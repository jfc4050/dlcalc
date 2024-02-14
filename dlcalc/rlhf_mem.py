"""CLI tool for various deep learning estimations."""

from argparse import ArgumentParser

from .memory.threed.states import States, ThreeDParallelModel, ParallelismConfig
from .utils.configurations import ActivationCheckpointingType


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "--actor-bparams",
        type=int,
        required=True,
        help="Actor/SFT number of parameters, in billions",
    )
    parser.add_argument("--actor-n-kv-heads", type=int, required=True)
    parser.add_argument("--actor-head-dim", type=int, required=True)
    parser.add_argument("--actor-n-layers", type=int, required=True)
    parser.add_argument("--actor-hidden-sz", type=int, required=True)

    parser.add_argument(
        "--critic-bparams",
        type=int,
        required=True,
        help="Critic/RM number of parameters, in billions",
    )
    parser.add_argument("--critic-n-kv-heads", type=int, required=True)
    parser.add_argument("--critic-head-dim", type=int, required=True)
    parser.add_argument("--critic-n-layers", type=int, required=True)
    parser.add_argument("--critic-hidden-sz", type=int, required=True)

    parser.add_argument("-s", "--sequence-len", type=int, required=True)
    parser.add_argument("-b", "--microbatch-sz", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--pp-critic", type=int, required=True)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--zero-level", type=int, required=True)
    parser.add_argument(
        "--activation-checkpointing-type",
        choices=["none", "full", "selective"],
        required=True,
    )
    args = parser.parse_args()
    print(args)
    print()

    actor_n_params = args.actor_bparams * 1e9
    actor_n_kv_heads = args.actor_n_kv_heads
    actor_head_dim = args.actor_head_dim
    actor_hidden_sz = args.actor_hidden_sz
    actor_n_layers = args.actor_n_layers

    critic_n_params = args.critic_bparams * 1e9
    critic_n_kv_heads = args.critic_n_kv_heads
    critic_head_dim = args.critic_head_dim
    critic_hidden_sz = args.critic_hidden_sz
    critic_n_layers = args.critic_n_layers

    sequence_len = args.sequence_len
    microbatch_sz = args.microbatch_sz
    tp_sz = args.tp
    pp_sz = args.pp
    pp_sz_critic = args.pp_critic
    dp_sz = 1  # TODO.
    zero_level = args.zero_level
    sequence_parallel = args.sequence_parallel
    activation_checkpointing_type = ActivationCheckpointingType.from_str(
        args.activation_checkpointing_type
    )

    actor_model = ThreeDParallelModel(
        n_params=actor_n_params,
        parallelism_cfg=ParallelismConfig(
            tp=tp_sz,
            pp=pp_sz,
            dp=dp_sz,
            sp_enabled=sequence_parallel,
            zero_level=ParallelismConfig.ZeroLevel(zero_level),
        ),
        sequence_len=sequence_len,
        microbatch_sz=microbatch_sz,
        hidden_sz=actor_hidden_sz,
        n_layers=actor_n_layers,
        n_kv_heads=actor_n_kv_heads,
        head_dim=actor_head_dim,
        activation_checkpointing_type=activation_checkpointing_type,
    )
    critic_model = ThreeDParallelModel(
        n_params=critic_n_params,
        parallelism_cfg=ParallelismConfig(
            tp=tp_sz,
            pp=pp_sz_critic,
            dp=dp_sz,
            sp_enabled=sequence_parallel,
            zero_level=ParallelismConfig.ZeroLevel(zero_level),
        ),
        sequence_len=sequence_len,
        microbatch_sz=microbatch_sz,
        hidden_sz=critic_hidden_sz,
        n_layers=critic_n_layers,
        n_kv_heads=critic_n_kv_heads,
        head_dim=critic_head_dim,
        activation_checkpointing_type=activation_checkpointing_type,
    )

    print("STATICS")
    print("--------------------------------------------------------------------------")
    all_states = []
    for model_name, model_def, training in [
        ("actor", actor_model, True),
        ("critic", critic_model, True),
        ("sft", actor_model, False),
        ("rm", critic_model, False),
    ]:
        print(f"{model_name}: frozen={int(not training)}")
        if training:
            states = States.for_unfrozen_3d_mixed_precision(model_def)
        else:
            states = States.for_frozen_3d_half_precision(model_def)

        all_states.append(states)
        print(states)
        print()

    print("TOTAL")
    print()

    # activations
    print("TRAINING ACTIVATIONS (ACTOR):")
    per_microbatch_per_layer = actor_model.activation_size_per_microbatch_per_layer()
    print("act/inflight_ubatch/layer:", per_microbatch_per_layer)

    layers_per_pp_stage = actor_model.layers_per_pp_stage()
    print(f"  layers/pp_stage = {actor_n_layers} / {pp_sz} = {layers_per_pp_stage}")

    per_microbatch = per_microbatch_per_layer * layers_per_pp_stage
    print("act/inflight_ubatch/pp_stage:", per_microbatch)

    max_inflight_microbatches = pp_sz

    total_act_memory = max_inflight_microbatches * per_microbatch
    print("act/pp_stage:", total_act_memory)


# if __name__ == "__main__":
#     main()
