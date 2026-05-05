from dlcalc.utils.configurations import ActivationCheckpointingType
from dlcalc.utils.model_3d import ParallelConfig, ThreeDParallelModel


def test_activation_breakdown_includes_partitioned_shapes() -> None:
    model = ThreeDParallelModel(
        parallelism_cfg=ParallelConfig(
            tp=8,
            cp=2,
            pp=1,
            dp=1,
            expert_mesh=None,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        ),
        sequence_len=2048,
        microbatch_sz=2,
        hidden_sz=4096,
        n_layers=32,
        n_q_heads=32,
        n_kv_heads=8,
        head_dim=128,
        inter_sz=11008,
        glu=True,
        moe_cfg=None,
        rotary_embed=True,
        dropout=True,
        vocab_sz=32000,
        tie_embeddings=True,
        act_ckpting_type=ActivationCheckpointingType.SELECTIVE,
        n_param_buckets=1,
    )

    activation_breakdown = model.activation_breakdown_per_microbatch_per_layer_detailed()

    assert activation_breakdown["Pre Attn Norm"].shape == "(1024, 2, 512)"
    assert activation_breakdown["Pre Attn Norm"].abstract_shape == "(s/cp, b, h/tp)"
    assert activation_breakdown["Query"].shape == "(1024, 2, 4, 128)"
    assert activation_breakdown["Query"].abstract_shape == "(s/cp, b, q/tp, d)"
    assert activation_breakdown["Up/Gate"].shape == "(1024, 2, 2752)"
    assert activation_breakdown["Up/Gate"].abstract_shape == "(s/cp, b, 2i/tp)"
    assert (
        activation_breakdown["Post Attention Dropout Mask"].shape == "(1024, 2, 512) (packed mask)"
    )
    assert (
        activation_breakdown["Post Attention Dropout Mask"].abstract_shape
        == "(s/cp, b, h/tp) (packed mask)"
    )
