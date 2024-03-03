"""CLI tool for various deep learning estimations."""

from argparse import ArgumentParser

from dlcalc.states import ThreeDParallelModel, ParallelismConfig
from dlcalc.utils.configurations import ActivationCheckpointingType


def main() -> None:
    parser = ArgumentParser(__doc__)
    # fmt: off
    parser.add_argument(
        "--bparams",
        type=int,
        required=True,
        help="Actor/SFT number of parameters, in billions",
    )
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
        action="store_true",
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
        choices=["none", "full", "selective"],
        required=True,
    )
    # fmt: on
    args = parser.parse_args()
    print(args)
    print()

    model_def = ThreeDParallelModel(
        n_params=args.bparams * 1e9,
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
        glu=True,  # TODO.
        vocab_sz=args.vocab_sz,
        activation_checkpointing_type=ActivationCheckpointingType.from_str(
            args.activation_checkpointing_type
        ),
    )

    print("STATICS")
    print("--------------------------------------------------------------------------")
    states = model_def.get_states(training=True)
    print(states)
    print("transformer block params")
    transformer_block_params = (
        model_def.layers_per_pp_stage() * model_def.get_transformer_block_n_params()
    )
    print(transformer_block_params)
    print("embedding/LM head params")
    print(model_def.get_embedding_or_lm_head_n_params())
    print("params (most loaded stages)")
    print(transformer_block_params + model_def.get_embedding_or_lm_head_n_params())
    print()

    # activations
    print("TRAINING ACTIVATIONS:")
    print("--------------------------------------------------------------------------")
    per_microbatch_per_layer_per_inflight = (
        model_def.activation_size_per_microbatch_per_layer()
    )
    print("inflight_ubatch_sz/layer:", per_microbatch_per_layer_per_inflight)

    max_inflight_microbatches = model_def.max_inflight_microbatches()
    print("max_inflight_microbatches:", max_inflight_microbatches)
    per_microbatch_per_layer = (
        per_microbatch_per_layer_per_inflight * max_inflight_microbatches
    )
    print("act/layer:", per_microbatch_per_layer)

    layers_per_pp_stage = model_def.layers_per_pp_stage()
    print(f"  layers/pp_stage = {layers_per_pp_stage}")

    per_microbatch = per_microbatch_per_layer * layers_per_pp_stage
    print("act/pp_stage:", per_microbatch)


if __name__ == "__main__":
    main()
