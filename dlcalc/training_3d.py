"""CLI tool for estimating performance characteristics of 3D parallel training."""

import json
import math
from argparse import ArgumentParser
from collections import OrderedDict

import yaml

from dlcalc.utils.comms import (
    get_all_to_all_comm_time_s,
    get_cross_dc_dp_all_gather_comm_time_s,
    get_cross_dc_dp_reduce_scatter_comm_time_s,
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
from dlcalc.utils.configurations import ActivationCheckpointingType, CrossDCConfig
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

    # Parse optional cross-DC configuration
    cross_dc_config = None
    if "cross_dc" in cfg and cfg["cross_dc"] is not None:
        cross_dc_config = CrossDCConfig(
            n_dcs=cfg["cross_dc"]["n_dcs"],
            interconnect_bandwidth_gbps=cfg["cross_dc"]["interconnect_bandwidth_gbps"],
            interconnect_latency_s=cfg["cross_dc"]["interconnect_latency_s"],
        )

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
        n_param_buckets=cfg["parallelism"]["n_param_buckets"],
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

    if cross_dc_config is not None:
        print_section_separator()
        print_info("Cross-DC Configuration")
        print_kv("Number of DCs", str(cross_dc_config.n_dcs), key_width=30)
        print_kv(
            "Interconnect Bandwidth",
            f"{cross_dc_config.interconnect_bandwidth_gbps:.0f} Gbps",
            key_width=30,
        )
        print_kv(
            "Interconnect Latency",
            f"{cross_dc_config.interconnect_latency_s * 1000:.2f} ms",
            key_width=30,
        )
        print_kv(
            "Max Ring Latency",
            f"{cross_dc_config.interconnect_latency_s * 1000:.2f} ms",
            key_width=30,
        )
        nodes_per_dc = safe_divide(cluster_size, machine_spec.n_devices) // cross_dc_config.n_dcs
        print_kv("Nodes per DC", str(nodes_per_dc), key_width=30)

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
    print_h1_header("MEMORY")
    print_section_separator()
    print_info("Model States")
    print(model_repr.states)

    print_section_separator()
    print_info("Activations")
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

    print()
    print_info("Activation Breakdown per Layer/Microbatch")
    activation_breakdown = model_repr.activation_breakdown_per_microbatch_per_layer()

    # Calculate max sizes for better alignment
    max_name_len = max(len(name) for name in activation_breakdown.keys())
    total_numel = sum(activation_breakdown.values())
    total_size_mib = (total_numel * model_repr.bits_per_parameter // 8) / (1024**2)

    for name, numel in activation_breakdown.items():
        size_mib = (numel * model_repr.bits_per_parameter // 8) / (1024**2)
        percentage = (numel / total_numel) * 100
        bar_width = int(percentage / 2)  # Scale to fit in terminal
        bar = "█" * bar_width if bar_width > 0 else ""
        color = get_color_for_component_percentage(percentage)
        print(
            f"    {name:<{max_name_len}} │ {numel:>12,} │ {size_mib:>8.1f} MiB │ {color} {percentage:>5.1f}% {bar}{_END}"
        )
    print(f"  {'─' * (max_name_len + 45)}")
    print(
        f"{_BOLD}    {'TOTAL':<{max_name_len}} │ {total_numel:>12,} │ {total_size_mib:>8.1f} MiB {_END}"
    )

    print_section_separator()
    print_info("Summary")
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

    # Initialize cross-DC communication times (will be updated if cross-DC is enabled)
    cross_dc_grad_bucket_rs_time_s = None
    cross_dc_param_bucket_ag_time_s = None

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
        n_buckets = model_repr.n_param_buckets
        mp_params_size = model_repr.states.params_shard.size(partitioned=True)
        param_bucket_numel = mp_params_size.numel() // model_repr.n_param_buckets
        # TODO. precisions here assume we are doing AMP
        param_bucket_size = Size(
            numel=param_bucket_numel,
            bits_per_element=model_repr.bits_per_parameter,
        )
        grad_bucket_size = Size(
            numel=param_bucket_numel,
            bits_per_element=model_repr.bits_per_grad,
        )

        print_kv("Params per MP rank", str(mp_params_size), key_width=30)
        print_kv("Bucket Size", f"{format_number(param_bucket_numel)} params", key_width=30)
        print_kv("Number of Buckets", str(n_buckets), key_width=30)
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

        # Cross-DC impact analysis
        if cross_dc_config is not None:
            print_section_separator()
            print_info("Cross-DC Impact on DP Communication")

            cross_dc_grad_bucket_rs_time_s = get_cross_dc_dp_reduce_scatter_comm_time_s(
                size=grad_bucket_size,
                parallel_config=model_repr.parallelism_cfg,
                machine_spec=machine_spec,
                cross_dc_config=cross_dc_config,
            )

            cross_dc_param_bucket_ag_time_s = get_cross_dc_dp_all_gather_comm_time_s(
                size=param_bucket_size,
                parallel_config=model_repr.parallelism_cfg,
                machine_spec=machine_spec,
                cross_dc_config=cross_dc_config,
            )

            rs_degradation_ms = (
                cross_dc_grad_bucket_rs_time_s - grad_bucket_reduce_scatter_time_s
            ) * 1000
            ag_degradation_ms = (
                cross_dc_param_bucket_ag_time_s - param_bucket_all_gather_time_s
            ) * 1000

            rs_degradation_pct = (
                rs_degradation_ms / (grad_bucket_reduce_scatter_time_s * 1000)
            ) * 100
            ag_degradation_pct = (ag_degradation_ms / (param_bucket_all_gather_time_s * 1000)) * 100

            print(f"\n  {_BOLD}Per-Bucket Cross-DC Degradation{_END}")
            print_kv(
                "  Reduce-Scatter Delta",
                f"{rs_degradation_ms:.3f} ms ({rs_degradation_pct:.1f}% slower)",
                key_width=30,
            )
            print_kv(
                "  All-Gather Delta",
                f"{ag_degradation_ms:.3f} ms ({ag_degradation_pct:.1f}% slower)",
                key_width=30,
            )

            total_cross_dc_rs_time = cross_dc_grad_bucket_rs_time_s * n_buckets * 1000
            total_cross_dc_ag_time = cross_dc_param_bucket_ag_time_s * n_buckets * 1000
            total_cross_dc_dp_time = total_cross_dc_rs_time + total_cross_dc_ag_time

            print(f"\n  {_BOLD}Total Cross-DC DP Communication{_END}")
            print_kv("  Cross-DC RS Total", f"{total_cross_dc_rs_time:.2f} ms", key_width=30)
            print_kv("  Cross-DC AG Total", f"{total_cross_dc_ag_time:.2f} ms", key_width=30)

            total_degradation_ms = total_cross_dc_dp_time - (total_rs_time + total_ag_time)
            total_degradation_pct = (total_degradation_ms / (total_rs_time + total_ag_time)) * 100

            print_metric(
                "  Total Cross-DC DP",
                f"{total_cross_dc_dp_time:.2f}",
                f"ms ({total_degradation_pct:.1f}% slower)",
                highlight=True,
            )

    ##################################################################################
    # Iteration Time
    ##################################################################################
    print_h1_header("PERFORMANCE: ITERATION TIME ANALYSIS")
    print_info(
        "NOTE: This is intended to give theoretical time estimates. \n  "
        "Any gaps between this and observations will be a combination of: \n  "
        "\n  "
        "a) errors in the modeling. please cut an issue if you find one. \n  "
        "b) implementation issues that you should consider fixing. \n  "
        "   common issues include CPU boundedness, jitter, stragglers, \n  "
        "   dataloading, etc. \n  "
    )
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
                n_local_experts
                * safe_divide(expert_capacity, model_repr.parallelism_cfg.expert_mesh.tp)
                * hidden_sz,
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
            {
                # Attention
                "Pre Attn Norm": hbm_load_store_time_s,  # norm approximation
                "RoPE": hbm_load_store_time_s,  # RoPE approximation
                "Pre Attn AG": ag_time_s,
                "QKV Proj": compute_gemm_time_s(model_repr.qkv_weight),
                "SDPA": sdpa_time,
                "Attn Out Proj": compute_gemm_time_s(model_repr.attn_out_weight),
                "Post Attn RS": rs_time_s,
                "Post Attn Residual": hbm_load_store_time_s,
                # MLP
                "Pre MLP Norm": hbm_load_store_time_s,  # norm approximation
                "Router": compute_gemm_time_s(model_repr.router_weight),
                # bunch of router operations that are hard to project and
                # are highly implementation dependent.
                "Pre MLP A2A": a2a_time_s,
                "Pre MLP AG": get_expert_tp_all_gather_comm_time_s(
                    size=expert_activation_size,
                    parallel_config=model_repr.parallelism_cfg,
                    machine_spec=machine_spec,
                ),
                "MLP Up Proj": compute_expert_gemm_time_s(
                    n_tokens_per_expert=expert_capacity,
                    weight_repr=model_repr.mlp_up_exp_weight,  # type: ignore[arg-type]
                ),
                "Glu Act": 3  # read 2, write 1
                * (
                    n_local_experts
                    * expert_capacity
                    * model_repr.moe_cfg.expert_inter_sz
                    * safe_divide(model_repr.bits_per_parameter, 8)
                )
                / machine_spec.device_spec.mem_bandwidth_bytes_per_sec,
                "MLP Down Proj": compute_expert_gemm_time_s(
                    n_tokens_per_expert=expert_capacity,
                    weight_repr=model_repr.mlp_down_exp_weight,  # type: ignore[arg-type]
                ),
                "Post MLP RS": get_expert_tp_reduce_scatter_comm_time_s(
                    size=expert_activation_size,
                    parallel_config=model_repr.parallelism_cfg,
                    machine_spec=machine_spec,
                ),
                "Post MLP A2A": a2a_time_s,
                "Post MLP Residual": hbm_load_store_time_s,
                "Activation Send": activation_send_time_s,
            }
        )
    else:
        transformer_block_time_components: dict[str, float] = OrderedDict(  # type: ignore[no-redef]
            {
                # Attention
                "Pre Attn Norm": hbm_load_store_time_s,  # norm approximation
                "RoPE": hbm_load_store_time_s,  # RoPE approximation
                "Pre Attn AG": ag_time_s,
                "QKV Proj": compute_gemm_time_s(model_repr.qkv_weight),
                "SDPA": sdpa_time,
                "Attn Out Proj": compute_gemm_time_s(model_repr.attn_out_weight),
                "Post Attn RS": rs_time_s,
                "Post Attn Residual": hbm_load_store_time_s,
                # MLP
                "Pre MLP Norm": hbm_load_store_time_s,  # norm approximation
                "Pre MLP AG": ag_time_s,
                "MLP Up Proj": compute_gemm_time_s(model_repr.mlp_up_weight),
                "MLP Down Proj": compute_gemm_time_s(model_repr.mlp_down_weight),
                "Post MLP RS": rs_time_s,
                "Post MLP Residual": hbm_load_store_time_s,
                "Activation Send": activation_send_time_s,
            }
        )

    print()
    print_h2_header("TRANSFORMER BLOCK COMPONENTS")
    total_transformer_block_time_s = sum(transformer_block_time_components.values())

    # Keep original ordering to preserve logical flow of operations
    for component_name, component_time_s in transformer_block_time_components.items():
        time_ms = component_time_s * 1000
        percentage = (component_time_s / total_transformer_block_time_s) * 100

        # Use color coding based on percentage (custom thresholds for block components)
        color = get_color_for_component_percentage(percentage)

        # Create a simple bar chart
        bar_length = int(percentage / 2)  # Scale to max 50 chars
        bar = "█" * bar_length

        print(
            f"  {component_name.ljust(25)} {color}{time_ms:7.2f} ms{_END}  {percentage:5.1f}%  {_GRAY}{bar}{_END}"
        )

    print()
    print_metric(
        "Total Block Time", f"{total_transformer_block_time_s * 1000:.2f}", "ms", highlight=True
    )
    print()

    transformer_block_fwd_time = sum(transformer_block_time_components.values())
    transformer_block_bwd_time = {
        ActivationCheckpointingType.NONE: 2 * transformer_block_fwd_time,
        ActivationCheckpointingType.SELECTIVE: 2 * transformer_block_fwd_time,
        ActivationCheckpointingType.SUPER_SELECTIVE: 2 * transformer_block_fwd_time,
        ActivationCheckpointingType.FULL: 3 * transformer_block_fwd_time,  # extra fwd
    }[model_repr.act_ckpting_type]
    n_fwds = n_microbatches_per_mp_rank * layers_per_pp_stage
    n_bwds = n_fwds

    transformer_block_time = (
        n_fwds * transformer_block_fwd_time + n_bwds * transformer_block_bwd_time
    )
    pipeline_bubble_time = transformer_block_time * pipeline_bubble_fraction

    # approximation: we'll assume all reductions are overlapped
    # except for at the first pipeline stage, where they are completely exposed.
    # Generally, for large training jobs where microbatch size is 1
    # (i.e. not much computation to overlap with), the first pipeline stage's DP
    # communication is almost completely exposed.
    # VPP doesn't help much with this - even though in theory it allows some DP
    # comms to be launched earlier, the DP costs are too expensive relative to final
    # microbatch costs for this to make a significant difference.
    # NOTE. this is an approximation when using EP, where there will be separate
    # reduction groups for DP_exp and DP_nonexp.
    if cross_dc_config is not None:
        assert cross_dc_param_bucket_ag_time_s is not None
        assert cross_dc_grad_bucket_rs_time_s is not None
        dp_ag_time = cross_dc_param_bucket_ag_time_s * n_buckets
        dp_rs_time = cross_dc_grad_bucket_rs_time_s * n_buckets
    else:
        dp_ag_time = param_bucket_all_gather_time_s * n_buckets
        dp_rs_time = grad_bucket_reduce_scatter_time_s * n_buckets

    # approximating optimizer step time as time to read/write states + optim states to/from HBM
    opt_step_time_s = (
        2
        * model_repr.states.total_bytes(partitioned=True)
        / machine_spec.device_spec.mem_bandwidth_bytes_per_sec
    )

    iteration_time_components: dict[str, float] = OrderedDict(
        {
            "Transformer Block": transformer_block_time,
            "DP All-Gather": dp_ag_time,
            "DP Reduce-Scatter": dp_rs_time,
            "Pipeline Bubble": pipeline_bubble_time,
            "Optimizer Step": opt_step_time_s,
        }
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

        # Use color coding based on percentage
        color = get_color_by_percentage(percentage)

        # Create a simple bar chart
        bar_length = int(percentage / 2)  # Scale to max 50 chars
        bar = "█" * bar_length

        print(
            f"  {component_name.ljust(20)} {color}{time_ms:8.2f} ms{_END}  {percentage:5.1f}%  {_GRAY}{bar}{_END}"
        )

    print()

    print_h2_header("FINAL RESULTS")
    print()
    print_metric("Iteration Time", f"{iteration_time_s:.2f}", "seconds", highlight=True)
    tokens_per_day = ((gbs * sequence_len) / iteration_time_s) * 60 * 60 * 24
    print_metric("Tokens per Day (w/ 100% Goodput)", f"{tokens_per_day / 1e9:.2f}", unit="B")

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
