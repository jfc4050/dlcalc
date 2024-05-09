"""CLI tool for estimating performance characteristics of 3D parallel training."""

import json
import math
from argparse import ArgumentParser

import yaml

from dlcalc.utils.comms import (
    get_dp_reduce_scatter_comm_time_s,
    get_dp_all_gather_comm_time_s,
    get_tp_all_gather_comm_time_s,
    get_tp_reduce_scatter_comm_time_s,
)
from dlcalc.utils.configurations import ActivationCheckpointingType
from dlcalc.utils.data import Size
from dlcalc.utils.hardware import MachineSpec
from dlcalc.utils.math import compute_gemm_flops, product, safe_divide
from dlcalc.utils.model_3d import ParallelConfig, ThreeDParallelModel
from dlcalc.utils.printing import print_section_header


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument("cfg_path", type=str)
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    print_section_header("CONFIG")
    print(json.dumps(cfg, indent=2))

    sequence_len = cfg["data"]["seqlen"]
    microbatch_sz = cfg["data"]["microbatch_sz"]
    hidden_sz = cfg["model"]["hidden_sz"]

    model_def = ThreeDParallelModel(
        parallelism_cfg=ParallelConfig(
            tp=cfg["parallelism"]["tp"],
            pp=cfg["parallelism"]["pp"],
            dp=cfg["parallelism"]["dp"],
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
    cluster_size = model_def.parallelism_cfg.world_size()
    print(machine_spec)
    print("n_devices: ", cluster_size)
    print("n_nodes: ", safe_divide(cluster_size, machine_spec.n_devices))

    ###################################################################################
    # MEMORY ANALYSIS
    ###################################################################################

    print_section_header("[MEMORY] STATES")
    print(f"total params: {model_def.get_total_n_params(partitioned=False) * 1e-9:.2f}B")
    print(model_def.states)

    # activations
    print_section_header("[MEMORY] TRAINING ACTIVATIONS")
    per_microbatch_per_layer_per_inflight = model_def.activation_size_per_microbatch_per_layer()
    print("act/layer/inflight:", per_microbatch_per_layer_per_inflight)
    max_inflight_microbatches = model_def.max_inflight_microbatches()
    layers_per_pp_stage = model_def.layers_per_pp_stage()
    vpp_penalty = model_def.vpp_penalty()
    print(f"VPP memory penalty = {vpp_penalty:.2f}")
    act_memory = (
        per_microbatch_per_layer_per_inflight
        * max_inflight_microbatches
        * math.ceil(vpp_penalty * layers_per_pp_stage)
    )
    print(
        f"act/pp_stage = "
        f"per_microbatch_per_layer_per_inflight * "
        f"{max_inflight_microbatches} * "
        f"{math.ceil(vpp_penalty * layers_per_pp_stage)} = "
        f"{act_memory}"
    )

    print_section_header("[MEMORY] TOTAL")
    print(
        f"total mem (GiB) = {(model_def.states.total_bytes(partitioned=True) + act_memory.bytes()) / (1024 ** 3):.3f}GiB"
    )

    ###################################################################################
    # PERF ANALYSIS
    ###################################################################################
    print_section_header("GEMMs (note numbers calculated for 100% flops+bandwidth utilization)")
    for proj_name, weight_repr in [
        ("QKV", model_def.qkv_weight),
        ("ATTN_OUT", model_def.attn_out_weight),
        ("MLP1", model_def.mlp_up_weight),
        ("MLP2", model_def.mlp_down_weight),
    ]:
        weight_partitioned_shape = weight_repr.shape(partitioned=True)
        weight_unpartitioned_shape = weight_repr.shape(partitioned=False)
        flops = compute_gemm_flops(
            weight_partitioned_shape,
            seqlen=model_def.sequence_len,
            batch_sz=model_def.microbatch_sz,
        )
        print(f"{proj_name} {weight_unpartitioned_shape} --tp--> {weight_partitioned_shape}")
        print(
            f"\tCOMPUTE: {flops * 1e-12:.2f} TFLOPs -> "
            f"{flops/machine_spec.device_spec.peak_flops * 1000:.3f} ms"
        )
        bytes_per_element = model_def.bits_per_parameter // 8
        gemm_input_dim, gemm_output_dim = weight_partitioned_shape
        weight_bytes = bytes_per_element * product(weight_partitioned_shape)
        input_bytes = bytes_per_element * product(
            (model_def.sequence_len, model_def.microbatch_sz, gemm_input_dim)
        )
        output_bytes = bytes_per_element * product(
            (model_def.sequence_len, model_def.microbatch_sz, gemm_output_dim)
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

    print_section_header("TP COMMUNICATION")
    # TODO. assumes SP, analysis pretty similar if not SP though
    activation_size = Size(
        numel=sequence_len * microbatch_sz * hidden_sz,
        bits_per_element=model_def.bits_per_parameter,
    )
    print(
        f"TP all-gather: {activation_size}: {get_tp_all_gather_comm_time_s(size=activation_size, n_participants=model_def.parallelism_cfg.tp, machine_spec=machine_spec) * 1000:.3f} ms"
    )
    print(
        f"TP reduce-scatter: {activation_size}: {get_tp_reduce_scatter_comm_time_s(size=activation_size, n_participants=model_def.parallelism_cfg.tp, machine_spec=machine_spec) * 1000:.3f} ms"
    )

    print_section_header("PIPELINE BUBBLE")
    gbs = cfg["data"]["gbs"]
    mbs = cfg["data"]["microbatch_sz"]
    vpp = cfg["parallelism"]["vpp"]
    bs_per_dp = safe_divide(gbs, model_def.parallelism_cfg.dp)
    n_microbatches = safe_divide(bs_per_dp, mbs)
    print(f"gbs = {gbs}")
    print(f"gbs/pipeline = {bs_per_dp}")
    print(f"VPP pipeline bubble multiplier = {(1 / vpp):.2f}")
    print(
        f"pipeline bubble fraction = {(1 / vpp) * (model_def.parallelism_cfg.pp - 1) / n_microbatches:.2f}"
    )

    print_section_header("DP COMMUNICATION")
    if model_def.parallelism_cfg.zero_level != ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER:
        raise NotImplementedError
    else:
        ###############################################################################
        # compute the backward time for a single microbatch.
        ###############################################################################
        # NOTE: this is fully sequential, there's no other microbatches to overlap with
        single_microbatch_bwd_flops = (
            2  # FLOPs/MAC
            * 2  # factor for backward only (2 GEMMs per op)
            * model_def.microbatch_sz
            * model_def.sequence_len
            * model_def.get_total_n_params(partitioned=True)
        )

        # divide by single pipeline stage TFLOPs, since its just for single
        # microbatch there's only one active pipeline stage at a time
        single_microbatch_bwd_time = single_microbatch_bwd_flops / machine_spec.total_flops()
        print(
            f"single MP rank, single microbatch bwd compute time {single_microbatch_bwd_time:.2f} s (if 100% FLOPs utilization)"
        )
        print()

        ###############################################################################
        # DP comm times
        ###############################################################################
        # grads are reduced in full-precision
        # params are all-gathered in half-precision
        mp_params_size = model_def.states.params_shard.size(partitioned=True)
        # TODO. precisions here assume we are doing AMP
        grad_bucket_size = model_def.grad_bucket_size()
        param_bucket_size = model_def.param_bucket_size()
        grad_bucket_reduce_scatter_time_s = get_dp_reduce_scatter_comm_time_s(
            size=grad_bucket_size,
            n_participants=model_def.parallelism_cfg.dp,
            machine_spec=machine_spec,
        )
        print(f"reduce_scatter(grad_bucket) time = {grad_bucket_reduce_scatter_time_s:.2f}s")
        param_bucket_all_gather_time_s = get_dp_all_gather_comm_time_s(
            size=param_bucket_size,
            n_participants=model_def.parallelism_cfg.dp,
            machine_spec=machine_spec,
        )
        print(f"all_gather(param_bucket) time = {param_bucket_all_gather_time_s:.2f}s")
        print()

        n_grad_buckets = int(math.ceil(mp_params_size.numel() / grad_bucket_size.numel()))
        n_param_buckets = int(math.ceil(mp_params_size.numel() / param_bucket_size.numel()))
        print(f"reduce_scatter n_buckets = {n_grad_buckets}")
        print(f"all_gather n_buckets = {n_param_buckets}")
        print()

        print(
            f"reduce_scatter(all_grads) time = {grad_bucket_reduce_scatter_time_s * n_grad_buckets:.2f}s"
        )
        print(
            f"all_gather(all_params) time = {param_bucket_all_gather_time_s * n_param_buckets:.2f}s"
        )

    print_section_header("WEAK SCALING")
    full_dp_comm_vol_factor = (model_def.parallelism_cfg.dp - 1) / model_def.parallelism_cfg.dp
    for dp in range(1, min(model_def.parallelism_cfg.dp, 8) + 1):
        factor = (dp - 1) / dp
        print(f"DP={dp} -> {(factor / full_dp_comm_vol_factor) * 100:.2f}% scaling degradation")
    print("...")


if __name__ == "__main__":
    main()
