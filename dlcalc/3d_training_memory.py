"""CLI tool for estimating memory consumption of 3D parallel training."""

from argparse import ArgumentParser
import json
import math

from dlcalc.states import ThreeDParallelModel, ParallelismConfig
from dlcalc.utils.configurations import ActivationCheckpointingType


def main() -> None:
    parser = ArgumentParser(__doc__)
    # fmt: off
    parser.add_argument(
        "--n-layers",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--hidden-sz",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--inter-sz",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--n-q-heads",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--n-kv-heads",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--vocab-sz",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--glu",
        type=bool,
        help="whether model uses gated linear units",
        required=True,
    )
    parser.add_argument(
        "--rotary-embeds",
        type=bool,
        help="whether model uses gated linear units",
        required=True,
    )
    parser.add_argument(
        "-s", "--sequence-len",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-b", "--microbatch-sz",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--tp",
        type=int,
        required=True,
        help="tensor model parallel (TP) degree",
    )
    parser.add_argument(
        "--pp",
        type=int,
        required=True,
        help="pipeline model parallel (PP) degree",
    )
    parser.add_argument(
        "--dp",
        type=int,
        required=True,
        help="data parallel (DP) degree",
    )
    parser.add_argument(
        "--vpp",
        type=int,
        required=True,
        help="virtual pipeline parallel (VPP) degree",
    )
    parser.add_argument(
        "--sequence-parallel",
        type=bool,
        required=True,
        help="whether or not sequence parallelism is enabled."
    )
    parser.add_argument(
        "--zero-level",
        type=int,
        required=True,
        choices=[0, 1, 2, 3],
        help="ZeRO partitioning level",
    )
    parser.add_argument(
        "--activation-checkpointing-type",
        choices=["none", "selective", "super-selective", "full"],
        required=True,
    )
    # fmt: on
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))
    print()

    model_def = ThreeDParallelModel(
        parallelism_cfg=ParallelismConfig(
            tp=args.tp,
            pp=args.pp,
            dp=args.dp,
            vpp=args.vpp,
            sp_enabled=args.sequence_parallel,
            zero_level=ParallelismConfig.ZeroLevel(args.zero_level),
        ),
        sequence_len=args.sequence_len,
        microbatch_sz=args.microbatch_sz,
        hidden_sz=args.hidden_sz,
        n_layers=args.n_layers,
        n_q_heads=args.n_q_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        inter_sz=args.inter_sz,
        glu=args.glu,
        rotary_embed=args.rotary_embeds,
        vocab_sz=args.vocab_sz,
        act_ckpting_type=ActivationCheckpointingType.from_str(
            args.activation_checkpointing_type
        ),
    )

    print("STATES")
    print("--------------------------------------------------------------------------")
    states = model_def.get_states(training=True)
    print(states)
    print()

    # activations
    print("TRAINING ACTIVATIONS:")
    print("--------------------------------------------------------------------------")
    per_microbatch_per_layer_per_inflight = (
        model_def.activation_size_per_microbatch_per_layer()
    )
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
        f"{per_microbatch_per_layer_per_inflight} * "
        f"{max_inflight_microbatches} * "
        f"{math.ceil(vpp_penalty * layers_per_pp_stage)} = "
        f"{act_memory}"
    )
    print()

    print("TOTAL:")
    print("--------------------------------------------------------------------------")
    print(
        f"total mem (GiB) = {(states.total_bytes() + act_memory.bytes()) / (1024 ** 3):.3f}GiB"
    )
    print()


if __name__ == "__main__":
    main()
