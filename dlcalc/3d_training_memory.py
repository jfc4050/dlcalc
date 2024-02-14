"""CLI tool for various deep learning estimations."""

from argparse import ArgumentParser

from dlcalc.states import States, ThreeDParallelModel, ParallelismConfig
from dlcalc.utils.configurations import ActivationCheckpointingType


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "--bparams",
        type=int,
        required=True,
        help="Actor/SFT number of parameters, in billions",
    )
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--hidden-sz", type=int, required=True)

    parser.add_argument("-s", "--sequence-len", type=int, required=True)
    parser.add_argument("-b", "--microbatch-sz", type=int, required=True)
    parser.add_argument(
        "--tp", type=int, required=True, help="tensor model parallel (TP) degree"
    )
    parser.add_argument(
        "--pp", type=int, required=True, help="pipeline model parallel (PP) degree"
    )
    parser.add_argument(
        "--dp", type=int, required=True, help="data parallel (DP) degree"
    )
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument(
        "--zero-level",
        type=int,
        required=True,
        choices=[0, 1, 2, 3],
        help="ZeRO partitioning level",
    )
    parser.add_argument(
        "--activation-checkpointing-type",
        choices=["none", "full", "selective"],
        required=True,
    )
    args = parser.parse_args()
    print(args)
    print()

    n_params = args.bparams * 1e9
    hidden_sz = args.hidden_sz
    n_layers = args.n_layers

    sequence_len = args.sequence_len
    microbatch_sz = args.microbatch_sz
    tp_sz = args.tp
    pp_sz = args.pp
    dp_sz = args.dp
    zero_level = args.zero_level
    sequence_parallel = args.sequence_parallel
    activation_checkpointing_type = ActivationCheckpointingType.from_str(
        args.activation_checkpointing_type
    )

    model_def = ThreeDParallelModel(
        n_params=n_params,
        parallelism_cfg=ParallelismConfig(
            tp=tp_sz,
            pp=pp_sz,
            dp=dp_sz,
            sp_enabled=sequence_parallel,
            zero_level=ParallelismConfig.ZeroLevel(zero_level),
        ),
        sequence_len=sequence_len,
        microbatch_sz=microbatch_sz,
        hidden_sz=hidden_sz,
        n_layers=n_layers,
        n_kv_heads=None,  # only needed for inference
        head_dim=None,  # only needed for inference
        activation_checkpointing_type=activation_checkpointing_type,
    )

    print("STATICS")
    print("--------------------------------------------------------------------------")
    states = States.for_unfrozen_3d_mixed_precision(model_def)
    print(states)
    print()

    # activations
    print("TRAINING ACTIVATIONS:")
    print("--------------------------------------------------------------------------")
    per_microbatch_per_layer_per_inflight = (
        model_def.activation_size_per_microbatch_per_layer()
    )
    print("act/layer/inflight_ubatch:", per_microbatch_per_layer_per_inflight)

    max_inflight_microbatches = pp_sz  # TODO.
    per_microbatch_per_layer = (
        per_microbatch_per_layer_per_inflight * max_inflight_microbatches
    )
    print("act/layer:", per_microbatch_per_layer)

    layers_per_pp_stage = model_def.layers_per_pp_stage()
    print(f"  layers/pp_stage = {n_layers} / {pp_sz} = {layers_per_pp_stage}")

    per_microbatch = per_microbatch_per_layer * layers_per_pp_stage
    print("act/pp_stage:", per_microbatch)


if __name__ == "__main__":
    main()
