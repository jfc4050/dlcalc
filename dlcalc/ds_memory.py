from dlcalc.states import States, Zero3Model

actor_n_params = 7 * 1e9
critic_n_params = 1.1 * 1e9
rm_n_params = 1.1 * 1e9

world_size = 8

actor_model_def = Zero3Model(n_params=actor_n_params, world_size=world_size)
critic_model_def = Zero3Model(n_params=critic_n_params, world_size=world_size)
rm_model_def = Zero3Model(n_params=rm_n_params, world_size=world_size)


print("STATICS")
print("--------------------------------------------------------------------------")
all_states = []
for model_name, model_def, training in [
    ("actor", actor_model_def, True),
    ("critic", critic_model_def, True),
    ("rm", rm_model_def, False),
]:
    print(f"{model_name}: frozen={int(not training)}")
    if training:
        states = States.for_unfrozen_zero3_half_precision(model_def)
    else:
        states = States.for_frozen_zero3_half_precision(model_def)

    all_states.append(states)
    print(states)
    print()

print("TOTAL")
print()


from torch.distributions import Categorical

Categorical