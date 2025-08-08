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
from dlcalc.utils.math import product, safe_divide
from dlcalc.utils.model_3d import MoeCfg, ParallelConfig, ThreeDParallelModel
from dlcalc.utils.printing import print_bold, print_h1_header, print_h2_header

ASSUMED_GEMM_UTIL = 0.7


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument("cfg_path", type=str)
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    print_h1_header("CONFIG")
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
    print(machine_spec)
    print("n_devices: ", cluster_size)
    print("n_nodes: ", safe_divide(cluster_size, machine_spec.n_devices))

    ###################################################################################
    # DATA
    ###################################################################################
    print_h1_header("DATA")
    gbs = cfg["data"]["gbs"]
    mbs = cfg["data"]["microbatch_sz"]

    bs_per_mp_rank = safe_divide(gbs, model_repr.parallelism_cfg.dp)
    n_microbatches_per_mp_rank = safe_divide(bs_per_mp_rank, mbs)

    print(f"GBS = {gbs} ({gbs * sequence_len * 1e-6:.2f}M tokens)")
    print(f"GBS/DP = {bs_per_mp_rank}")
    print(f"n_microbatches = {n_microbatches_per_mp_rank}")

    ###################################################################################
    # MODEL SUMMARY
    ###################################################################################
    print_h1_header("MODEL SUMMARY")
    print(
        f"params: {model_repr.get_n_total_params(partitioned=False) * 1e-9:.2f}B "
        f"({model_repr.get_n_active_params(partitioned=False) * 1e-9:.2f}B active)"
    )

    ###################################################################################
    # MEMORY ANALYSIS
    ###################################################################################
    print_h1_header("[MEMORY] STATES")
    print(model_repr.states)

    # activations
    print_h1_header("[MEMORY] TRAINING ACTIVATIONS")
    act_size_per_layer_per_inflight_microbatch = (
        model_repr.activation_size_per_microbatch_per_layer()
    )
    print("act/layer/inflight:", act_size_per_layer_per_inflight_microbatch)
    max_inflight_microbatches = model_repr.parallelism_cfg.pp  # 1F1B
    print("max(inflight):", max_inflight_microbatches)
    layers_per_pp_stage = model_repr.layers_per_pp_stage()
    print("layers/pp:", layers_per_pp_stage)
    vpp_multiplier = model_repr.vpp_penalty()
    print(f"VPP memory multiplier = {vpp_multiplier:.2f}")
    act_memory = (
        act_size_per_layer_per_inflight_microbatch
        * min(n_microbatches_per_mp_rank, max_inflight_microbatches)
        * math.ceil(vpp_multiplier * layers_per_pp_stage)
    )
    print(
        f"act/pp_stage = "
        f"per_microbatch_per_layer_per_inflight * "
        f"{max_inflight_microbatches} * "
        f"{math.ceil(vpp_multiplier * layers_per_pp_stage)} = "
        f"{act_memory}"
    )

    print_h1_header("[MEMORY] TOTAL")
    print(
        f"total mem (GiB) = {(model_repr.states.total_bytes(partitioned=True) + act_memory.bytes()) / (1024**3):.3f}GiB"
    )

    ###################################################################################
    # PERF ANALYSIS
    ###################################################################################
    print_h1_header("GEMMs (note numbers calculated for 100% flops+bandwidth utilization)")
    projections = OrderedDict(
        {
            "QKV": model_repr.qkv_weight,
            "ATTN_OUT": model_repr.attn_out_weight,
            "MLP1": model_repr.mlp_up_weight,
            "MLP2": model_repr.mlp_down_weight,
            # TODO. need different section for MoE
        }
    )
    if model_repr.mlp_up_exp_weight is not None:
        expert_dim, *other_dims = model_repr.mlp_up_exp_weight.shape(partitioned=False)
        single_expert_shape = tuple(other_dims)
        single_expert_partition_spec = {
            k - 1: v for k, v in model_repr.mlp_up_exp_weight._partition_spec.items() if k != 0
        }

        projections["MLP1_EXP"] = TensorRepr(
            unpartitioned_shape=single_expert_shape,
            partition_spec=single_expert_partition_spec,
            bits_per_elt=model_repr.bits_per_parameter,
        )
    if model_repr.mlp_down_exp_weight is not None:
        expert_dim, *other_dims = model_repr.mlp_down_exp_weight.shape(partitioned=False)
        single_expert_shape = tuple(other_dims)
        single_expert_partition_spec = {
            k - 1: v for k, v in model_repr.mlp_down_exp_weight._partition_spec.items() if k != 0
        }

        projections["MLP2_EXP"] = TensorRepr(
            unpartitioned_shape=single_expert_shape,
            partition_spec=single_expert_partition_spec,
            bits_per_elt=model_repr.bits_per_parameter,
        )

    for proj_name, weight_repr in projections.items():
        weight_repr: TensorRepr  # type: ignore[no-redef]
        flops = compute_gemm_flops(
            n_tokens=safe_divide(model_repr.sequence_len, model_repr.parallelism_cfg.cp)
            * model_repr.microbatch_sz,
            weight_shape=weight_repr.shape(partitioned=True),
        )
        print(
            f"{proj_name} {weight_repr.shape(partitioned=False)} --tp--> {weight_repr.shape(partitioned=True)}"
        )
        print(
            f"\tCOMPUTE: {flops * 1e-12:.2f} TFLOPs -> "
            f"{flops / machine_spec.device_spec.peak_flops * 1000:.3f} ms"
        )
        bytes_per_element = model_repr.bits_per_parameter // 8
        gemm_input_dim, gemm_output_dim = weight_repr.shape(partitioned=True)
        weight_bytes = bytes_per_element * weight_repr.numel(partitioned=True)
        input_bytes = bytes_per_element * product(
            safe_divide(model_repr.sequence_len, model_repr.parallelism_cfg.cp),
            model_repr.microbatch_sz,
            gemm_input_dim,
        )
        output_bytes = bytes_per_element * product(
            safe_divide(model_repr.sequence_len, model_repr.parallelism_cfg.cp),
            model_repr.microbatch_sz,
            gemm_output_dim,
        )
        print(
            f"\tLOAD INPUT: {input_bytes * 1e-9:.2f} GB -> "
            f"{input_bytes / (machine_spec.device_spec.mem_bandwidth_bytes_per_sec) * 1000:.3f} ms"
        )
        print(
            f"\tLOAD_WEIGHT: {weight_bytes * 1e-9:.2f} GB -> "
            f"{weight_bytes / (machine_spec.device_spec.mem_bandwidth_bytes_per_sec) * 1000:.3f} ms"
        )
        print(
            f"\tSTORE_OUTPUT: {output_bytes * 1e-9:.2f} GB -> "
            f"{output_bytes / (machine_spec.device_spec.mem_bandwidth_bytes_per_sec) * 1000:.3f} ms"
        )

    print_h1_header("TP COMMUNICATION")
    # TODO. assumes SP, analysis pretty similar if not SP though
    activation_size = Size(
        numel=safe_divide(sequence_len, model_repr.parallelism_cfg.cp) * microbatch_sz * hidden_sz,
        bits_per_element=model_repr.bits_per_parameter,
    )
    print(
        f"TP all-gather: {activation_size}: {get_tp_all_gather_comm_time_s(size=activation_size, parallel_config=model_repr.parallelism_cfg, machine_spec=machine_spec) * 1000:.3f} ms"
    )
    print(
        f"TP reduce-scatter: {activation_size}: {get_tp_reduce_scatter_comm_time_s(size=activation_size, parallel_config=model_repr.parallelism_cfg, machine_spec=machine_spec) * 1000:.3f} ms"
    )

    print_h1_header("PP COMMUNICATION")
    activation_send_time_s = (
        activation_size.bytes() / machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec
    )
    print(f"PP send/recv: {activation_size}: {activation_send_time_s * 1000:.3f} ms")

    print_h1_header("PIPELINE BUBBLE")

    vpp = cfg["parallelism"]["vpp"]
    bs_per_mp_rank = safe_divide(gbs, model_repr.parallelism_cfg.dp)
    n_microbatches_per_mp_rank = safe_divide(bs_per_mp_rank, mbs)
    pipeline_bubble_fraction = (
        (1 / vpp) * (model_repr.parallelism_cfg.pp - 1) / n_microbatches_per_mp_rank
    )

    print(f"VPP pipeline bubble multiplier = {(1 / vpp):.2f}")
    print(f"pipeline bubble fraction = {pipeline_bubble_fraction:.2f}")

    print_h1_header("DP COMMUNICATION")
    if model_repr.parallelism_cfg.zero_level != ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER:
        raise NotImplementedError
    else:
        # TODO. we need to fix these to account for EP and CP.
        ###############################################################################
        # compute the backward time for a single microbatch.
        ###############################################################################
        devices_in_pp_stage_flops = (
            model_repr.parallelism_cfg.tp * machine_spec.device_spec.peak_flops
        )

        # divide by single pipeline stage TFLOPs, since its just for single
        # microbatch there's only one active pipeline stage at a time
        single_microbatch_fwd_time = (
            model_repr.get_single_microbatch_fwd_flops() / devices_in_pp_stage_flops
        )
        single_microbatch_bwd_time = (
            model_repr.get_single_microbatch_bwd_flops() / devices_in_pp_stage_flops
        )
        print(
            f"single PP rank: (if 100% FLOPs utilization):\n"
            f"* single microbatch fwd compute time {single_microbatch_fwd_time * 1000:.3f} ms\n"
            f"* single microbatch bwd compute time {single_microbatch_bwd_time * 1000:.3f} ms"
        )
        print()

        ###############################################################################
        # DP comm times
        ###############################################################################
        # grads are reduced in full-precision
        # params are all-gathered in half-precision
        mp_params_size = model_repr.states.params_shard.size(partitioned=True)
        print(f"params per MP degree: {mp_params_size}")
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
        print(f"reduce_scatter/all_gather n_buckets = ceil({n_buckets})")
        print()
        # full BW should be divided along all MP ranks within a single node, since
        # they are each participating in their own DP collectives. We make the
        # assumption here that TP is the only form of MP we do within node.
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
        print(
            f"reduce_scatter(1_grad_bucket):\n"
            f"\tlatency term = {grad_bucket_reduce_scatter_lat_term_s * 1000:.3f} ms\n"
            f"\tbw term = {grad_bucket_reduce_scatter_bw_term_s * 1000:.3f} ms (if 100% BW utilization)\n"
            f"\tTOTAL = {grad_bucket_reduce_scatter_time_s * 1000:.3f} ms\n"
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
        print(
            f"all_gather(1_param_bucket):\n"
            f"\tlatency term = {param_bucket_all_gather_lat_term_s * 1000:.3f} ms\n"
            f"\tbw term = {param_bucket_all_gather_bw_term_s * 1000:.3f} ms (if 100% BW utilization)\n"
            f"\tTOTAL = {param_bucket_all_gather_time_s * 1000:.3f} ms\n"
        )

        print(
            f"reduce_scatter(all_grad_buckets) time = {grad_bucket_reduce_scatter_time_s * n_buckets * 1000:.3f} ms "
            f"(if 100% BW utilization)"
        )
        print(
            f"all_gather(all_param_buckets) time = {param_bucket_all_gather_time_s * n_buckets * 1000:.3f} ms "
            f"(if 100% BW utilization)"
        )

    ##################################################################################
    # Iteration Time
    ##################################################################################
    print_h1_header("ITERATION TIME (IN PROGRESS - DON'T TRUST ME)")
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
    n_heads_per_tp_partition = model_repr.n_q_heads // model_repr.parallelism_cfg.tp
    sdpa_flops = sum(
        [
            # Q @ K.T
            2 * n_heads_per_tp_partition * sequence_len * model_repr.head_dim * sequence_len,
            # A @ V
            2 * n_heads_per_tp_partition * sequence_len * sequence_len * model_repr.head_dim,
        ]
    )
    sdpa_time = (sdpa_flops // model_repr.parallelism_cfg.cp) / (
        machine_spec.device_spec.peak_flops * ASSUMED_GEMM_UTIL
    )

    if model_repr.moe_cfg is not None:
        assert model_repr.parallelism_cfg.expert_mesh is not None

        expert_capacity = (
            int(
                model_repr.microbatch_sz
                * model_repr.sequence_len
                * model_repr.moe_cfg.experts_per_token  # or k in other words
                * model_repr.parallelism_cfg.expert_mesh.ep
                * model_repr.moe_cfg.capacity_factor
            )
            // model_repr.moe_cfg.n_experts
        )
        n_local_experts = safe_divide(
            model_repr.moe_cfg.n_experts,
            model_repr.parallelism_cfg.expert_mesh.ep,
        )

        print(f"expert capacity: {expert_capacity}")

        def compute_expert_gemm_time_s(n_tokens_per_expert: int, weight_repr: TensorRepr) -> float:
            n_local_experts, *gemm_dims = weight_repr.shape(partitioned=True)
            flops = n_local_experts * compute_gemm_flops(
                n_tokens=n_tokens_per_expert,
                weight_shape=tuple(gemm_dims),
            )
            return flops / (machine_spec.device_spec.peak_flops * ASSUMED_GEMM_UTIL)

        # the alltoall will trade token partitioning along the capacity dimension
        # for token partitioning along the experts dimension. i.e. we'll go from:
        # (capacity / prod(token_partitioning_degrees_exp) / EP, n_experts) to
        # (capacity / prod(token_partitioning_degrees_exp), n_experts / EP)
        a2a_time_s = get_all_to_all_comm_time_s(
            size=Size(
                expert_capacity * hidden_sz,
                bits_per_element=model_repr.bits_per_parameter,
            ),
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )

        expert_activation_size = Size(
            numel=n_local_experts * expert_capacity * model_repr.hidden_sz,
            bits_per_element=model_repr.bits_per_parameter,
        )

        expert_ag_time_s = get_expert_tp_all_gather_comm_time_s(
            size=expert_activation_size,
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )
        expert_rs_time_s = get_expert_tp_reduce_scatter_comm_time_s(
            size=expert_activation_size,
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
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
            pre_mlp_ag=expert_ag_time_s,
            mlp_up_proj=compute_expert_gemm_time_s(
                n_tokens_per_expert=expert_capacity,
                weight_repr=model_repr.mlp_up_exp_weight,  # type: ignore[arg-type]
            ),
            # TODO. GLU Activation
            mlp_down_proj=compute_expert_gemm_time_s(
                n_tokens_per_expert=expert_capacity,
                weight_repr=model_repr.mlp_down_exp_weight,  # type: ignore[arg-type]
            ),
            post_mlp_rs=expert_rs_time_s,
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
    for component_name, component_time_s in transformer_block_time_components.items():
        print(
            component_name.ljust(30),
            f"{component_time_s * 1000:.2f}ms / {(component_time_s / total_transformer_block_time_s) * 100:.2f}%",
        )

    print(f"total transformer block: {total_transformer_block_time_s * 1000:.2f}ms")
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
    for component_name, component_time_s in iteration_time_components.items():
        print(
            component_name.ljust(30),
            f"{component_time_s * 1000:.2f}ms / {(component_time_s / iteration_time_s) * 100:.2f}%",
        )
    print()

    print_h2_header("SUMMARY")
    print_bold(f"iteration time: {iteration_time_s:.2f}s")

    ideal_iteration_time = (
        gbs
        * sequence_len
        * 6
        * model_repr.get_n_active_params(partitioned=False)
        / (cluster_size * machine_spec.device_spec.peak_flops)
    )
    print_bold(f"ideal iteration time: {ideal_iteration_time:.5f}s")

    print_bold(f"predicted MFU: {(ideal_iteration_time / iteration_time_s) * 100:.2f}%")


if __name__ == "__main__":
    main()
