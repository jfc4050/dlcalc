"""CLI tool for estimating performance characteristics of 3D parallel training."""

import json
import math
from argparse import ArgumentParser

import yaml

from dlcalc.utils.comms import get_reduce_scatter_comm_volume
from dlcalc.utils.configurations import ActivationCheckpointingType
from dlcalc.utils.hardware import MachineSpec
from dlcalc.utils.math import compute_gemm_tflops, safe_divide
from dlcalc.utils.model_3d import ParallelConfig, ThreeDParallelModel


def _print_section_header(section_name: str) -> None:
    print()
    print("--------------------------------------------------------------------------")
    print(section_name)
    print("--------------------------------------------------------------------------")
    pass


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument("cfg_path", type=str)
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    _print_section_header("CONFIG")
    print(json.dumps(cfg, indent=2))

    model_def = ThreeDParallelModel(
        parallelism_cfg=ParallelConfig(
            tp=cfg["parallelism"]["tp"],
            pp=cfg["parallelism"]["pp"],
            dp=cfg["parallelism"]["dp"],
            vpp=cfg["parallelism"]["vpp"],
            sp_enabled=cfg["parallelism"]["sp"],
            zero_level=ParallelConfig.ZeroLevel(cfg["parallelism"]["zero_level"]),
        ),
        sequence_len=cfg["data"]["seqlen"],
        microbatch_sz=cfg["data"]["microbatch_sz"],
        hidden_sz=cfg["model"]["hidden_sz"],
        n_layers=cfg["model"]["n_layers"],
        n_q_heads=cfg["model"]["n_q_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        head_dim=cfg["model"]["head_dim"],
        inter_sz=cfg["model"]["inter_sz"],
        glu=cfg["model"]["glu"],
        rotary_embed=cfg["model"]["rotary_embeds"],
        vocab_sz=cfg["model"]["vocab_sz"],
        tie_embeddings=cfg["model"]["tie_embeddings"],
        act_ckpting_type=ActivationCheckpointingType.from_str(
            cfg["performance"]["activation_checkpointing_type"]
        ),
    )
    machine_spec = MachineSpec.from_str(cfg["hardware"]["node_type"])
    cluster_size = model_def.parallelism_cfg.world_size()
    print(machine_spec)
    print("n_devices: ", cluster_size)
    print("n_nodes: ", safe_divide(cluster_size, machine_spec.n_devices))

    ###################################################################################
    # MEMORY ANALYSIS
    ###################################################################################

    _print_section_header("STATES")
    print(f"total params: {model_def.get_total_n_params(partitioned=False) * 1e-9:.2f}B")
    states = model_def.get_partitioned_states(training=True)
    print(states)

    # activations
    _print_section_header("TRAINING ACTIVATIONS:")
    per_microbatch_per_layer_per_inflight = model_def.activation_size_per_microbatch_per_layer()
    print("act/layer/inflight:", per_microbatch_per_layer_per_inflight)
    max_inflight_microbatches = model_def.max_inflight_microbatches()
    layers_per_pp_stage = model_def.layers_per_pp_stage()
    vpp_penalty = model_def.vpp_penalty()
    print(f"VPP memory penalty: {vpp_penalty:.2f}")
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

    _print_section_header("TOTAL MEM")
    print(
        f"total mem (GiB) = {(states.total_bytes(partitioned=True) + act_memory.bytes()) / (1024 ** 3):.3f}GiB"
    )

    ###################################################################################
    # PERF ANALYSIS
    ###################################################################################
    _print_section_header("GEMMs")
    for proj_name, proj_shape in [
        ("QKV", model_def.qkv_weight.shape(partitioned=False)),
        ("attn_out", model_def.attn_out_weight.shape(partitioned=False)),
        ("MLP1", model_def.mlp_up_weight.shape(partitioned=False)),
        ("MLP2", model_def.mlp_down_weight.shape(partitioned=False)),
    ]:
        tflops = compute_gemm_tflops(
            proj_shape,
            seqlen=model_def.sequence_len,
            batch_sz=model_def.microbatch_sz,
        )
        print(
            f"{proj_name} ({proj_shape}): {tflops:.2f}TFLOPs -> "
            f"{tflops/(model_def.parallelism_cfg.tp * machine_spec.device_spec.peak_tflops) * 1000:.3f}ms expected runtime "
            f"(with 100% FLOPs utilization)"
        )

    _print_section_header("PIPELINE BUBBLE")
    gbs = cfg["data"]["gbs"]
    mbs = cfg["data"]["microbatch_sz"]
    vpp = cfg["parallelism"]["vpp"]
    bs_per_dp = safe_divide(gbs, model_def.parallelism_cfg.dp)
    n_microbatches = safe_divide(bs_per_dp, mbs)
    print(f"gbs={gbs}")
    print(f"gbs/dp={bs_per_dp}")
    print(f"VPP pipeline bubble multiplier={(1 / vpp):.2f}")
    print(
        f"pipeline bubble fraction: {(1 / vpp) * (model_def.parallelism_cfg.pp - 1) / n_microbatches:.2f}"
    )

    _print_section_header("DP COMM")
    if model_def.parallelism_cfg.zero_level != ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER:
        raise NotImplementedError
    else:
        # compute the backward time for a single microbatch.
        # NOTE: this is fully sequential, there's no other microbatches to overlap with
        single_microbatch_bwd_tflops = (
            2  # FLOPs/MAC
            * 2  # factor for backward only (2 GEMMs per op)
            * model_def.microbatch_sz
            * model_def.sequence_len
            * model_def.get_total_n_params(partitioned=False)
        ) * 1e-12

        # divide by single pipeline stage TFLOPs, since its just for single
        # microbatch there's only one active pipeline stage at a time
        single_microbatch_bwd_time = single_microbatch_bwd_tflops / machine_spec.total_flops()
        print(
            f"single microbatch bwd compute time {single_microbatch_bwd_time:.2f}s (with 100% FLOPs utilization)"
        )

        grad_size = states.opt_states.grad_buffer.size(partitioned=True)
        grad_reduce_scatter_vol = get_reduce_scatter_comm_volume(
            size=grad_size, n_participants=model_def.parallelism_cfg.dp
        )
        # NOTE: assumes duplex = 2x unidirectional
        grad_reduce_scatter_time_s = grad_reduce_scatter_vol.bits() / (
            # divide full BW among devices (which will be part of different DP groups)
            # TODO. this is a safe assumption vast majority of the time but might
            # want to handle when this isn't the case.
            machine_spec.inter_node_connect.unidirectional_bw_bits_per_sec / machine_spec.n_devices
        )
        print(f"reduce_scatter(grads) vol: {grad_reduce_scatter_vol}")
        print(f"reduce_scatter(grads) time: {grad_reduce_scatter_time_s:.2f}s")

    _print_section_header("WEAK SCALING")
    full_dp_comm_vol_factor = (model_def.parallelism_cfg.dp - 1) / model_def.parallelism_cfg.dp
    for dp in range(1, min(model_def.parallelism_cfg.dp, 8) + 1):
        factor = (dp - 1) / dp
        print(f"DP={dp} -> {(factor / full_dp_comm_vol_factor) * 100:.2f}% scaling degradation")
    print("...")


if __name__ == "__main__":
    main()
