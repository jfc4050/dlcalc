"""CLI tool for estimating memory consumption of 3D parallel training."""

from argparse import ArgumentParser
import json
import math

import yaml

from dlcalc.utils.states import ThreeDParallelModel, ParallelConfig
from dlcalc.utils.comms import get_grad_reducescatter_volume
from dlcalc.utils.configurations import ActivationCheckpointingType
from dlcalc.utils.hardware import MachineSpec


def _print_section_header(section_name: str) -> None:
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
    print()

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
        act_ckpting_type=ActivationCheckpointingType.from_str(
            cfg["performance"]["activation_checkpointing_type"]
        ),
    )
    machine_spec = MachineSpec.from_str(cfg["hardware"]["node_type"])
    print(machine_spec)

    ###################################################################################
    # MEMORY ANALYSIS
    ###################################################################################

    _print_section_header("STATES")
    print("total params: ", model_def.get_total_n_params())
    print()
    states = model_def.get_states(training=True)
    print(states)
    print()

    # activations
    _print_section_header("TRAINING ACTIVATIONS:")
    per_microbatch_per_layer_per_inflight = model_def.activation_size_per_microbatch_per_layer()
    print("act/layer/inflight:", per_microbatch_per_layer_per_inflight)
    max_inflight_microbatches = model_def.max_inflight_microbatches()
    layers_per_pp_stage = model_def.layers_per_pp_stage()
    vpp_penalty = model_def.vpp_penalty()
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
    print()

    _print_section_header("TOTAL MEM")
    print(f"total mem (GiB) = {(states.total_bytes() + act_memory.bytes()) / (1024 ** 3):.3f}GiB")
    print()

    ###################################################################################
    # PERF ANALYSIS
    ###################################################################################
    # TODO. pipeline bubble
    if model_def.parallelism_cfg.zero_level != ParallelConfig.ZeroLevel.PARTITION_GRADIENTS:
        raise NotImplementedError
    else:
        param_size = states.params

        grad_size = states.grads
        dp_size = model_def.parallelism_cfg.dp

        # compute the backward time for a single microbatch.
        # NOTE: this is fully sequential, there's no other microbatches to overlap with
        single_microbatch_bwd_tflops = (
            2  # FLOPs/MAC
            * 2  # factor for backward only
            * model_def.microbatch_sz
            * model_def.sequence_len
            * model_def.get_total_n_params().numel()
        ) * 1e-12

        # divide by single pipeline stage TFLOPs
        single_microbatch_bwd_time = single_microbatch_bwd_tflops / machine_spec.total_flops()
        print(f"single microbatch_bwd_time {single_microbatch_bwd_time:.2f}s")

        grad_reduce_scatter_vol = get_grad_reducescatter_volume(
            grad_size=grad_size, dp_size=dp_size
        )
        # NOTE: assumes duplex = 2x unidirectional
        grad_reduce_scatter_time_s = grad_reduce_scatter_vol.bits() / (
            # divide full BW among devices
            machine_spec.inter_node_connect.unidirectional_bw_bps / machine_spec.n_devices
        )
        print(f"reduce_scatter(grads) vol: {grad_reduce_scatter_vol}")
        print(f"reduce_scatter(grads) time: {grad_reduce_scatter_time_s}")


if __name__ == "__main__":
    main()
