"""CLI tool for estimating performance characteristics of 3D parallel training."""

import json
import math
from argparse import ArgumentParser
from collections import OrderedDict

import yaml

from dlcalc.utils.comms import (
    get_all_to_all_comm_time_s,
    get_dp_all_gather_bw_term_s,
    get_dp_all_gather_comm_time_s,
    get_dp_all_gather_latency_term_s,
    get_dp_reduce_scatter_bw_term_s,
    get_dp_reduce_scatter_comm_time_s,
    get_dp_reduce_scatter_latency_term_s,
    get_expert_tp_all_gather_comm_time_s,
    get_expert_tp_reduce_scatter_comm_time_s,
    get_tp_all_gather_comm_time_s,
    get_tp_reduce_scatter_comm_time_s,
)
from dlcalc.utils.compute import compute_gemm_flops
from dlcalc.utils.configurations import ActivationCheckpointingType
from dlcalc.utils.data import Size, TensorRepr
from dlcalc.utils.hardware import MachineSpec
from dlcalc.utils.math import safe_divide
from dlcalc.utils.model_3d import MoeCfg, ParallelConfig, ThreeDParallelModel
from dlcalc.utils.printing import (
    _BOLD,
    _END,
    _GRAY,
    format_number,
    get_color_by_percentage,
    get_color_by_time_ms,
    get_color_for_component_percentage,
    print_h1_header,
    print_h2_header,
    print_info,
    print_kv,
    print_metric,
    print_section_separator,
    print_success,
)

ASSUMED_GEMM_UTIL = 0.7


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument("cfg_path", type=str)
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    print_h1_header("CONFIGURATION")
    print(json.dumps(cfg, indent=2))

    sequence_len = cfg["data"]["seqlen"]
    microbatch_sz = cfg["data"]["microbatch_sz"]
    hidden_sz = cfg["model"]["hidden_sz"]

    # Setup expert parallelism configuration if MoE is enabled
    expert_mesh = None
    if "moe" in cfg["model"]:
        ep = cfg["parallelism"]["ep"]
        expert_tp = cfg["model"]["moe"]["expert_tp_degree"]
        # Calculate expert_dp from the constraint
        tp = cfg["parallelism"]["tp"]
        cp = cfg["parallelism"].get("cp", 1)
        dp = cfg["parallelism"]["dp"]
        expert_dp = safe_divide(dp * cp * tp, ep * expert_tp)
        expert_mesh = ParallelConfig.ExpertParallelCfg(ep=ep, tp=expert_tp, dp=expert_dp)
        print("expert mesh:", expert_mesh)

    model_repr = ThreeDParallelModel(
        parallelism_cfg=ParallelConfig(
            tp=cfg["parallelism"]["tp"],
            cp=cfg["parallelism"].get("cp", 1),
            pp=cfg["parallelism"]["pp"],
            dp=cfg["parallelism"]["dp"],
            expert_mesh=expert_mesh,
            vpp=cfg["parallelism"]["vpp"],
            sp_enabled=cfg["parallelism"]["sp"],
            zero_level=ParallelConfig.ZeroLevel(cfg["parallelism"]["zero_level"]),
        ),
        sequence_len=sequence_len,
        microbatch_sz=microbatch_sz,
        hidden_sz=hidden_sz,
        n_layers=cfg["model"]["n_layers"],
        n_q_heads=cfg["model"]["n_q_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        head_dim=cfg["model"]["head_dim"],
        inter_sz=cfg["model"]["inter_sz"],
        glu=cfg["model"]["glu"],
        moe_cfg=MoeCfg(
            n_experts=cfg["model"]["moe"]["n_experts"],
            expert_inter_sz=cfg["model"]["moe"]["expert_inter_sz"],
            experts_per_token=cfg["model"]["moe"]["experts_per_token"],
            capacity_factor=cfg["model"]["moe"]["capacity_factor"],
            moe_frequency=cfg["model"]["moe"]["moe_frequency"],
            expert_tp_degree=cfg["model"]["moe"]["expert_tp_degree"],
        )
        if "moe" in cfg["model"]
        else None,
        rotary_embed=cfg["model"]["rotary_embeds"],
        dropout=cfg["model"]["dropout"],
        vocab_sz=cfg["model"]["vocab_sz"],
        tie_embeddings=cfg["model"]["tie_embeddings"],
        act_ckpting_type=ActivationCheckpointingType.from_str(
            cfg["performance"]["activation_checkpointing_type"]
        ),
        bucket_size_bytes=int(cfg["parallelism"]["bucket_size_mb"] * 1e6),
    )

    machine_spec = MachineSpec.from_str(cfg["hardware"]["node_type"])
    cluster_size = model_repr.parallelism_cfg.world_size()

    print_section_separator()
    print_info("Hardware Configuration")
    print_kv("Node Type", cfg["hardware"]["node_type"], key_width=30)
    print_kv("Total Devices", str(cluster_size), key_width=30)
    print_kv("Total Nodes", str(safe_divide(cluster_size, machine_spec.n_devices)), key_width=30)
    print_kv(
        "Device Memory",
        f"{machine_spec.device_spec.mem_capacity_bytes / (1024**3):.0f} GiB",
        key_width=30,
    )
    print_kv(
        "Peak FLOPS/device",
        f"{machine_spec.device_spec.peak_flops / 1e12:.0f} TFLOPS",
        key_width=30,
    )

    ###################################################################################
    # DATA
    ###################################################################################
    print_section_separator()
    print_info("Data Configuration")
    gbs = cfg["data"]["gbs"]
    mbs = cfg["data"]["microbatch_sz"]

    bs_per_mp_rank = safe_divide(gbs, model_repr.parallelism_cfg.dp)
    n_microbatches_per_mp_rank = safe_divide(bs_per_mp_rank, mbs)

    print_kv("Global Batch Size", f"{gbs} samples", key_width=30)
    print_kv("Total Tokens/Batch", f"{format_number(gbs * sequence_len)} tokens", key_width=30)
    print_kv("Batch Size per DP Rank", str(bs_per_mp_rank), key_width=30)
    print_kv("Microbatches per Rank", str(n_microbatches_per_mp_rank), key_width=30)
    print_kv("Sequence Length", f"{sequence_len} tokens", key_width=30)

    ###################################################################################
    # MODEL SUMMARY
    ###################################################################################
    print_section_separator()
    print_info("Model Architecture")
    total_params = model_repr.get_n_total_params(partitioned=False)
    active_params = model_repr.get_n_active_params(partitioned=False)

    print_metric("Total Parameters", format_number(total_params), highlight=True)
    print_metric("Active Parameters", format_number(active_params))
    print_kv("Hidden Size", str(hidden_sz), key_width=30)
    print_kv("Number of Layers", str(cfg["model"]["n_layers"]), key_width=30)
    print_kv(
        "Attention Heads",
        f"{cfg['model']['n_q_heads']} (Q) / {cfg['model']['n_kv_heads']} (KV)",
        key_width=30,
    )

    ###################################################################################
    # MEMORY ANALYSIS
    ###################################################################################
    print_h1_header("MEMORY: MODEL STATES")
    print(model_repr.states)

    # activations
    print_h1_header("MEMORY: TRAINING ACTIVATIONS")
    act_size_per_layer_per_inflight_microbatch = (
        model_repr.activation_size_per_microbatch_per_layer()
    )
    max_inflight_microbatches = model_repr.parallelism_cfg.pp  # 1F1B
    layers_per_pp_stage = model_repr.layers_per_pp_stage()
    vpp_multiplier = model_repr.vpp_penalty()

    print_kv(
        "Activation/Layer/Microbatch", str(act_size_per_layer_per_inflight_microbatch), key_width=30
    )
    print_kv("Max Inflight Microbatches", str(max_inflight_microbatches), key_width=30)
    print_kv("Layers per PP Stage", str(layers_per_pp_stage), key_width=30)
    print_kv("VPP Memory Multiplier", f"{vpp_multiplier:.2f}x", key_width=30)
    act_memory = (
        act_size_per_layer_per_inflight_microbatch
        * min(n_microbatches_per_mp_rank, max_inflight_microbatches)
        * math.ceil(vpp_multiplier * layers_per_pp_stage)
    )
    print_kv("Total Activation Memory", f"{act_memory.bytes() / (1024**3):.3f} GiB", key_width=30)

    print_h1_header("MEMORY: SUMMARY")
    total_memory_gib = (model_repr.states.total_bytes(partitioned=True) + act_memory.bytes()) / (
        1024**3
    )
    print_success(f"Total Memory Required: {total_memory_gib:.3f} GiB per device")

    ###################################################################################
    # PERF ANALYSIS
    ###################################################################################
    print_h1_header("COMPUTE: GEMM OPERATIONS")
    print_info("Note: Numbers calculated assuming 100% FLOPS and bandwidth utilization")
    n_tokens_cp = (
        safe_divide(model_repr.sequence_len, model_repr.parallelism_cfg.cp)
        * model_repr.microbatch_sz
    )
    projections = OrderedDict(
        {
            "QKV Projection": (model_repr.qkv_weight, n_tokens_cp),
            "Attention Combine Projection": (model_repr.attn_out_weight, n_tokens_cp),
            "MLP Up Projection": (model_repr.mlp_up_weight, n_tokens_cp),
            "MLP Down Projection": (model_repr.mlp_down_weight, n_tokens_cp),
        }
    )

    if model_repr.mlp_up_exp_weight is not None:
        expert_dim, *other_dims = model_repr.mlp_up_exp_weight.shape(partitioned=False)
        single_expert_shape = tuple(other_dims)
        single_expert_partition_spec = {
            k - 1: v for k, v in model_repr.mlp_up_exp_weight._partition_spec.items() if k != 0
        }

        projections["MLP Up (Expert)"] = (
            TensorRepr(
                unpartitioned_shape=single_expert_shape,
                partition_spec=single_expert_partition_spec,
                bits_per_elt=model_repr.bits_per_parameter,
            ),
            model_repr.expert_capacity(),
        )
    if model_repr.mlp_down_exp_weight is not None:
        expert_dim, *other_dims = model_repr.mlp_down_exp_weight.shape(partitioned=False)
        single_expert_shape = tuple(other_dims)
        single_expert_partition_spec = {
            k - 1: v for k, v in model_repr.mlp_down_exp_weight._partition_spec.items() if k != 0
        }

        projections["MLP Down (Expert)"] = (
            TensorRepr(
                unpartitioned_shape=single_expert_shape,
                partition_spec=single_expert_partition_spec,
                bits_per_elt=model_repr.bits_per_parameter,
            ),
            model_repr.expert_capacity(),
        )

    for proj_name, (weight_repr, n_tokens) in projections.items():
        flops = compute_gemm_flops(
            n_tokens=n_tokens,
            weight_shape=weight_repr.shape(partitioned=True),
        )
        compute_time_ms = flops / machine_spec.device_spec.peak_flops * 1000

        # Color based on compute intensity
        color = get_color_by_time_ms(compute_time_ms)

        print(f"\n  {_BOLD}{proj_name}{_END}")
        print(
            f"    Shape: {weight_repr.shape(partitioned=False)} → Partitioned: {weight_repr.shape(partitioned=True)}"
        )

        # Compute metrics with bar
        print(f"    Compute: {color}{compute_time_ms:.3f} ms{_END}")
        print(f"             {_GRAY}({format_number(float(flops))} FLOPs){_END}")

        # Memory bandwidth metrics in a compact format
        bytes_per_element = safe_divide(model_repr.bits_per_parameter, 8)
        gemm_input_dim, gemm_output_dim = weight_repr.shape(partitioned=True)
        weight_bytes = bytes_per_element * weight_repr.numel(partitioned=True)
        input_bytes = n_tokens * gemm_input_dim * bytes_per_element
        output_bytes = n_tokens * gemm_output_dim * bytes_per_element
        input_time_ms = input_bytes / machine_spec.device_spec.mem_bandwidth_bytes_per_sec * 1000
        weight_time_ms = weight_bytes / machine_spec.device_spec.mem_bandwidth_bytes_per_sec * 1000
        output_time_ms = output_bytes / machine_spec.device_spec.mem_bandwidth_bytes_per_sec * 1000
        print(f"    Memory:  Input: {input_bytes / 1e9:.2f} GB ({input_time_ms:.3f} ms)")
        print(f"             Weight: {weight_bytes / 1e9:.2f} GB ({weight_time_ms:.3f} ms)")
        print(f"             Output: {output_bytes / 1e9:.2f} GB ({output_time_ms:.3f} ms)")

    print()

    print_h1_header("COMMUNICATION: TENSOR PARALLELISM")
    if not model_repr.parallelism_cfg.sp_enabled:
        raise NotImplementedError("not implemented for non-SP case")

    activation_size = Size(
        numel=safe_divide(sequence_len, model_repr.parallelism_cfg.cp) * microbatch_sz * hidden_sz,
        bits_per_element=model_repr.bits_per_parameter,
    )
    tp_ag_time = get_tp_all_gather_comm_time_s(
        size=activation_size, parallel_config=model_repr.parallelism_cfg, machine_spec=machine_spec
    )
    tp_rs_time = get_tp_reduce_scatter_comm_time_s(
        size=activation_size, parallel_config=model_repr.parallelism_cfg, machine_spec=machine_spec
    )

    print_kv("TP All-Gather", f"{tp_ag_time * 1000:.3f} ms", key_width=30)
    print_kv("TP Reduce-Scatter", f"{tp_rs_time * 1000:.3f} ms", key_width=30)
    print_kv("Activation Size", str(activation_size), key_width=30)

    print_h1_header("COMMUNICATION: PIPELINE PARALLELISM")
    activation_send_time_s = (
        activation_size.bytes() / machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec
    )
    print_kv("PP Send/Recv Time", f"{activation_send_time_s * 1000:.3f} ms", key_width=30)
    print_kv("Activation Size", str(activation_size), key_width=30)

    print_h1_header("PERFORMANCE: PIPELINE BUBBLE")

    vpp = cfg["parallelism"]["vpp"]
    bs_per_mp_rank = safe_divide(gbs, model_repr.parallelism_cfg.dp)
    n_microbatches_per_mp_rank = safe_divide(bs_per_mp_rank, mbs)
    pipeline_bubble_fraction = (
        (1 / vpp) * (model_repr.parallelism_cfg.pp - 1) / n_microbatches_per_mp_rank
    )

    print_kv("VPP Pipeline Bubble Multiplier", f"{(1 / vpp):.2f}x", key_width=30)
    print_kv("Pipeline Bubble Fraction", f"{pipeline_bubble_fraction:.2%}", key_width=30)

    print_h1_header("COMMUNICATION: DATA PARALLELISM")
    if model_repr.parallelism_cfg.zero_level != ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER:
        raise NotImplementedError
    else:
        print_section_separator()
        print_info("Microbatch Compute Times (100% FLOPS utilization)")

        devices_in_pp_stage_flops = (
            model_repr.parallelism_cfg.cp
            * model_repr.parallelism_cfg.tp
            * machine_spec.device_spec.peak_flops
        )

        # divide by single pipeline stage TFLOPs, since its just for single
        # microbatch there's only one active pipeline stage at a time
        single_microbatch_fwd_time = (
            model_repr.get_single_microbatch_fwd_flops() / devices_in_pp_stage_flops
        )
        single_microbatch_bwd_time = (
            model_repr.get_single_microbatch_bwd_flops() / devices_in_pp_stage_flops
        )

        print_kv("Forward Pass", f"{single_microbatch_fwd_time * 1000:.3f} ms", key_width=30)
        print_kv("Backward Pass", f"{single_microbatch_bwd_time * 1000:.3f} ms", key_width=30)

        # Gradient bucketing configuration
        print_section_separator()
        print_info("Gradient Bucketing")

        # grads are reduced in full-precision
        # params are all-gathered in half-precision
        mp_params_size = model_repr.states.params_shard.size(partitioned=True)
        # TODO. precisions here assume we are doing AMP
        grad_bucket_numel = model_repr.grad_bucket_numel()
        grad_bucket_size = Size(
            numel=grad_bucket_numel,
            bits_per_element=model_repr.bits_per_grad,
        )
        param_bucket_size = Size(
            numel=grad_bucket_numel,
            bits_per_element=model_repr.bits_per_parameter,
        )
        n_buckets = mp_params_size.numel() / grad_bucket_numel

        print_kv("Params per MP rank", str(mp_params_size), key_width=30)
        print_kv("Bucket Size", f"{format_number(grad_bucket_numel)} params", key_width=30)
        print_kv("Number of Buckets", f"{math.ceil(n_buckets)}", key_width=30)
        grad_bucket_reduce_scatter_lat_term_s = get_dp_reduce_scatter_latency_term_s(
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )
        grad_bucket_reduce_scatter_bw_term_s = get_dp_reduce_scatter_bw_term_s(
            grad_bucket_size,
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )
        grad_bucket_reduce_scatter_time_s = get_dp_reduce_scatter_comm_time_s(
            size=grad_bucket_size,
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )
        # Communication breakdown per bucket
        print_section_separator()
        print_info("Communication Breakdown (per bucket)")

        print(f"\n  {_BOLD}Reduce-Scatter (Gradients){_END}")
        print_kv(
            "  Attributed to Latency",
            f"{grad_bucket_reduce_scatter_lat_term_s * 1000:.3f} ms",
            key_width=30,
        )
        print_kv(
            "  Attributed to Bandwidth",
            f"{grad_bucket_reduce_scatter_bw_term_s * 1000:.3f} ms",
            key_width=30,
        )
        print_metric(
            "  Total", f"{grad_bucket_reduce_scatter_time_s * 1000:.3f}", "ms", highlight=True
        )

        param_bucket_all_gather_lat_term_s = get_dp_all_gather_latency_term_s(
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )
        param_bucket_all_gather_bw_term_s = get_dp_all_gather_bw_term_s(
            param_bucket_size,
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )
        param_bucket_all_gather_time_s = get_dp_all_gather_comm_time_s(
            param_bucket_size,
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )

        print(f"\n  {_BOLD}All-Gather (Parameters){_END}")
        print_kv(
            "  Attributed to Latency",
            f"{param_bucket_all_gather_lat_term_s * 1000:.3f} ms",
            key_width=30,
        )
        print_kv(
            "  Attributed to Bandwidth",
            f"{param_bucket_all_gather_bw_term_s * 1000:.3f} ms",
            key_width=30,
        )
        print_metric(
            "  Total", f"{param_bucket_all_gather_time_s * 1000:.3f}", "ms", highlight=True
        )

        # Total communication times
        print_section_separator()
        print_info("Total Communication Time (all buckets)")

        total_rs_time = grad_bucket_reduce_scatter_time_s * n_buckets * 1000
        total_ag_time = param_bucket_all_gather_time_s * n_buckets * 1000

        print_kv("Reduce-Scatter Total", f"{total_rs_time:.2f} ms", key_width=30)
        print_kv("All-Gather Total", f"{total_ag_time:.2f} ms", key_width=30)
        print_metric(
            "Combined DP Comm", f"{total_rs_time + total_ag_time:.2f}", "ms", highlight=True
        )

    ##################################################################################
    # Iteration Time
    ##################################################################################
    print_h1_header("PERFORMANCE: ITERATION TIME ANALYSIS")
    n_tokens = model_repr.microbatch_sz * model_repr.sequence_len

    def compute_gemm_time_s(weight_repr: TensorRepr) -> float:
        flops = compute_gemm_flops(
            n_tokens=safe_divide(n_tokens, model_repr.parallelism_cfg.cp),
            weight_shape=weight_repr.shape(partitioned=True),
        )
        return flops / (machine_spec.device_spec.peak_flops * ASSUMED_GEMM_UTIL)

    ag_time_s = get_tp_all_gather_comm_time_s(
        size=activation_size,
        parallel_config=model_repr.parallelism_cfg,
        machine_spec=machine_spec,
    )
    rs_time_s = get_tp_reduce_scatter_comm_time_s(
        size=activation_size,
        parallel_config=model_repr.parallelism_cfg,
        machine_spec=machine_spec,
    )

    hbm_load_store_time_s = (
        2 * activation_size.bytes() / machine_spec.device_spec.mem_bandwidth_bytes_per_sec
    )

    # SDPA time
    sdpa_flops = safe_divide(model_repr.n_q_heads, model_repr.parallelism_cfg.tp) * sum(
        [
            # Q @ K.T
            2
            * safe_divide(sequence_len, model_repr.parallelism_cfg.cp)
            * model_repr.head_dim
            * sequence_len,
            # A @ V
            2 * sequence_len * sequence_len * model_repr.head_dim,
        ]
    )
    sdpa_time = sdpa_flops / (machine_spec.device_spec.peak_flops * ASSUMED_GEMM_UTIL)

    if model_repr.moe_cfg is not None:
        assert model_repr.parallelism_cfg.expert_mesh is not None

        expert_capacity = model_repr.expert_capacity()
        n_local_experts = safe_divide(
            model_repr.moe_cfg.n_experts,
            model_repr.parallelism_cfg.expert_mesh.ep,
        )

        def compute_expert_gemm_time_s(n_tokens_per_expert: int, weight_repr: TensorRepr) -> float:
            n_local_experts, *gemm_dims = weight_repr.shape(partitioned=True)
            print(
                f"n_local_experts: {n_local_experts} n_tokens: {n_tokens_per_expert} gemm_dims: {gemm_dims}"
            )
            flops = n_local_experts * compute_gemm_flops(
                n_tokens=n_tokens_per_expert,
                weight_shape=tuple(gemm_dims),
            )
            return flops / (machine_spec.device_spec.peak_flops * ASSUMED_GEMM_UTIL)

        a2a_time_s = get_all_to_all_comm_time_s(
            size=Size(
                n_local_experts * expert_capacity * hidden_sz,
                bits_per_element=model_repr.bits_per_parameter,
            ),
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )

        expert_activation_size = Size(
            numel=n_local_experts * expert_capacity * model_repr.hidden_sz,
            bits_per_element=model_repr.bits_per_parameter,
        )

        transformer_block_time_components: dict[str, float] = OrderedDict(
            # Attention
            pre_attn_norm=hbm_load_store_time_s,  # norm approximation
            rope=hbm_load_store_time_s,  # RoPE approximation
            pre_attn_ag=ag_time_s,
            qkv_proj=compute_gemm_time_s(model_repr.qkv_weight),
            sdpa=sdpa_time,
            attn_out_proj=compute_gemm_time_s(model_repr.attn_out_weight),
            post_attn_rs=rs_time_s,
            post_attn_residual=hbm_load_store_time_s,
            # MLP
            pre_mlp_norm=hbm_load_store_time_s,  # norm approximation
            router=compute_gemm_time_s(model_repr.router_weight),
            # bunch of router operations that are hard to project and
            # are highly implementation dependent.
            pre_mlp_a2a=a2a_time_s,
            pre_mlp_ag=get_expert_tp_all_gather_comm_time_s(
                size=expert_activation_size,
                parallel_config=model_repr.parallelism_cfg,
                machine_spec=machine_spec,
            ),
            mlp_up_proj=compute_expert_gemm_time_s(
                n_tokens_per_expert=expert_capacity,
                weight_repr=model_repr.mlp_up_exp_weight,  # type: ignore[arg-type]
            ),
            glu_act=3  # read 2, write 1
            * (
                n_local_experts
                * expert_capacity
                * model_repr.moe_cfg.expert_inter_sz
                * safe_divide(model_repr.bits_per_parameter, 8)
            )
            / machine_spec.device_spec.mem_bandwidth_bytes_per_sec,
            mlp_down_proj=compute_expert_gemm_time_s(
                n_tokens_per_expert=expert_capacity,
                weight_repr=model_repr.mlp_down_exp_weight,  # type: ignore[arg-type]
            ),
            post_mlp_rs=get_expert_tp_reduce_scatter_comm_time_s(
                size=expert_activation_size,
                parallel_config=model_repr.parallelism_cfg,
                machine_spec=machine_spec,
            ),
            post_mlp_a2a=a2a_time_s,
            post_mlp_residual=hbm_load_store_time_s,
            activation_send=activation_send_time_s,
        )
    else:
        transformer_block_time_components: dict[str, float] = OrderedDict(  # type: ignore[no-redef]
            # Attention
            pre_attn_norm=hbm_load_store_time_s,  # norm approximation
            rope=hbm_load_store_time_s,  # RoPE approximation
            pre_attn_ag=ag_time_s,
            qkv_proj=compute_gemm_time_s(model_repr.qkv_weight),
            sdpa=sdpa_time,
            attn_out_proj=compute_gemm_time_s(model_repr.attn_out_weight),
            post_attn_rs=rs_time_s,
            post_attn_residual=hbm_load_store_time_s,
            # MLP
            pre_mlp_norm=hbm_load_store_time_s,  # norm approximation
            pre_mlp_ag=ag_time_s,
            mlp_up_proj=compute_gemm_time_s(model_repr.mlp_up_weight),
            mlp_down_proj=compute_gemm_time_s(model_repr.mlp_down_weight),
            post_mlp_rs=rs_time_s,
            post_mlp_residual=hbm_load_store_time_s,
            activation_send=activation_send_time_s,
        )

    print()
    print_h2_header("TRANSFORMER BLOCK COMPONENTS")
    total_transformer_block_time_s = sum(transformer_block_time_components.values())

    # Keep original ordering to preserve logical flow of operations
    for component_name, component_time_s in transformer_block_time_components.items():
        time_ms = component_time_s * 1000
        percentage = (component_time_s / total_transformer_block_time_s) * 100

        # Format component name with proper spacing
        formatted_name = component_name.replace("_", " ").title()

        # Use color coding based on percentage (custom thresholds for block components)
        color = get_color_for_component_percentage(percentage)

        # Create a simple bar chart
        bar_length = int(percentage / 2)  # Scale to max 50 chars
        bar = "█" * bar_length

        print(
            f"  {formatted_name.ljust(25)} {color}{time_ms:7.2f} ms{_END}  {percentage:5.1f}%  {_GRAY}{bar}{_END}"
        )

    print()
    print_metric(
        "Total Block Time", f"{total_transformer_block_time_s * 1000:.2f}", "ms", highlight=True
    )
    print()

    transformer_block_fwd_time = sum(transformer_block_time_components.values())
    transformer_block_bwd_time = 2 * transformer_block_fwd_time
    n_fwds = n_microbatches_per_mp_rank * layers_per_pp_stage
    n_bwds = n_fwds

    transformer_block_time = (
        n_fwds * transformer_block_fwd_time + n_bwds * transformer_block_bwd_time
    )
    pipeline_bubble_time = transformer_block_time * pipeline_bubble_fraction

    dp_ag_time = param_bucket_all_gather_time_s * n_buckets
    dp_rs_time = grad_bucket_reduce_scatter_time_s * n_buckets

    # approximating optimizer step time as time to read/write states + optim states to/from HBM
    opt_step_time_s = (
        2
        * model_repr.states.total_bytes(partitioned=True)
        / machine_spec.device_spec.mem_bandwidth_bytes_per_sec
    )

    iteration_time_components: dict[str, float] = OrderedDict(
        dp_ag=dp_ag_time,
        transformer_block=transformer_block_time,
        pipeline_bubble=pipeline_bubble_time,
        dp_rs=dp_rs_time,
        opt_step=opt_step_time_s,
    )

    print_h2_header("ITERATION TIME COMPONENTS")
    iteration_time_s = sum(iteration_time_components.values())

    # Sort components by time (descending) for better readability
    sorted_iteration_components = sorted(
        iteration_time_components.items(), key=lambda x: x[1], reverse=True
    )

    for component_name, component_time_s in sorted_iteration_components:
        time_ms = component_time_s * 1000
        percentage = (component_time_s / iteration_time_s) * 100

        # Format component name
        name_map = {
            "dp_ag": "DP All-Gather",
            "dp_rs": "DP Reduce-Scatter",
            "transformer_block": "Transformer Blocks",
            "pipeline_bubble": "Pipeline Bubble",
            "opt_step": "Optimizer Step",
        }
        formatted_name = name_map.get(component_name, component_name.replace("_", " ").title())

        # Use color coding based on percentage
        color = get_color_by_percentage(percentage)

        # Create a simple bar chart
        bar_length = int(percentage / 2)  # Scale to max 50 chars
        bar = "█" * bar_length

        print(
            f"  {formatted_name.ljust(20)} {color}{time_ms:8.2f} ms{_END}  {percentage:5.1f}%  {_GRAY}{bar}{_END}"
        )

    print()

    print_h2_header("FINAL RESULTS")
    print()
    print_metric("Iteration Time", f"{iteration_time_s:.2f}", "seconds", highlight=True)

    ideal_iteration_time = (
        gbs
        * sequence_len
        * 6
        * model_repr.get_n_active_params(partitioned=False)
        / (cluster_size * machine_spec.device_spec.peak_flops)
    )
    print_metric("Ideal Iteration Time", f"{ideal_iteration_time:.5f}", "seconds")

    mfu_percentage = (ideal_iteration_time / iteration_time_s) * 100
    print()
    print_success(f"Theoretical MFU: {mfu_percentage:.2f}%")
    print()


if __name__ == "__main__":
    main()
