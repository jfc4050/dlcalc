import dataclasses
from enum import Enum
from typing import Optional

from .configurations import ActivationCheckpointingType
from .data import Size, TensorRepr
from .math import ceil_divide, safe_divide


@dataclasses.dataclass
class ParallelConfig:
    class ZeroLevel(int, Enum):
        NONE = 0
        PARTITION_OPTIMIZER = 1
        PARTITION_GRADIENTS = 2
        PARTITION_PARAMETERS = 3

    @dataclasses.dataclass
    class ExpertParallelCfg:
        ep: int  # Expert Parallel (EP) degree
        tp: int
        dp: int

    tp: int  # Tensor Parallel (TP) degree
    cp: int  # Context Parallel (CP) degree
    pp: int  # Pipeline Parallel (PP) degree
    dp: int  # Data Parallel (DP) degree

    expert_mesh: ExpertParallelCfg | None

    vpp: int  # Virtual Pipeline Parallel (VPP) degree

    sp_enabled: bool  # Sequence Parallel (SP) enablement

    zero_level: ZeroLevel

    def __post_init__(self) -> None:
        if self.expert_mesh is not None:
            assert (
                self.expert_mesh.ep * self.expert_mesh.tp * self.expert_mesh.dp
                == self.dp * self.cp * self.tp
            )

    def world_size(self) -> int:
        return self.tp * self.cp * self.pp * self.dp


class DistributedAdamOptimizerStates:
    """Optimizer states from Apex DistributedFusedAdam.
    see: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md
    and: https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/distributed_fused_adam.py

    the distributed optimizer recieves a set of parameters to manage that are already
    model parallel partitioned. Internally, it will additionally partition states
    over DP, so optimizer states end up being partitioned over MP * DP, but
    the diestributed optimizer doesn't have any concept of model parallelism.

    NOTE: shards will be larger in reality due to alignment requirements and
    unfilled buckets.
    """

    # NOTE: n_params here is meant to be the parameters in a pipeline stage.
    def __init__(self, n_params: int, store_param_remainders: bool, dp: int) -> None:
        self.param_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            # apex has a optimization to avoid storing information that's redundant
            # between fp32 and fp16 weights. it can instead store an extra 16
            # bits of precision, which can be combined with bf16 weights to yield
            # fp32 weights
            bits_per_elt=16 if store_param_remainders else 32,
            enforce_evenly_partitionable=False,
        )
        self.exp_avg_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )
        self.exp_avg_sq_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )
        # reused for various purposes but at one point it needs to hold
        # all (i.e. not DP partitioned) parameter gradients, which are accumulated
        # to until they are reduce-scattered to a gradient shard during the final
        # microbatch.
        # basically ZeRO2, but with no reduce-scatter between gradient accumulations.
        self.grad_buffer = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=32,
        )

    def total_bytes(self, partitioned: bool) -> int:
        return _sum(
            self.param_shard.size(partitioned=partitioned).bytes(),
            self.exp_avg_shard.size(partitioned=partitioned).bytes(),
            self.exp_avg_sq_shard.size(partitioned=partitioned).bytes(),
            self.grad_buffer.size(partitioned=partitioned).bytes(),
        )


@dataclasses.dataclass
class ModelStates:
    """Tracks persistent tensor allocations kept throughput training."""

    params_shard: TensorRepr
    opt_states: DistributedAdamOptimizerStates

    def total_bytes(self, partitioned: bool) -> int:
        return self.params_shard.size(
            partitioned=partitioned
        ).bytes() + self.opt_states.total_bytes(partitioned=partitioned)

    def __repr__(self) -> str:
        params_bytes = self.params_shard.size(partitioned=True).bytes()
        opt_states_bytes = self.opt_states.total_bytes(partitioned=True)

        return "\n".join(
            [
                f"params          : {self.params_shard.size(partitioned=True)}",
                f"params (opt)    : {self.opt_states.param_shard.size(partitioned=True)}",
                f"exp_avg         : {self.opt_states.exp_avg_shard.size(partitioned=True)}",
                f"exp_avg_squared : {self.opt_states.exp_avg_sq_shard.size(partitioned=True)}",
                f"grad_buffer     : {self.opt_states.grad_buffer.size(partitioned=True)}",
                f"TOTAL           : {(params_bytes + opt_states_bytes) / (1024**3):.2f}GiB",
            ]
        )


def _sum(*summands: int) -> int:
    return sum(list(summands))


@dataclasses.dataclass
class MoeCfg:
    n_experts: int
    expert_inter_sz: int
    experts_per_token: int
    capacity_factor: float
    moe_frequency: float
    expert_tp_degree: int


@dataclasses.dataclass
class ThreeDParallelModel:
    """Representation of a 3D parallel transformer model."""

    parallelism_cfg: ParallelConfig

    sequence_len: int
    microbatch_sz: int

    hidden_sz: int

    n_layers: int

    n_q_heads: int
    n_kv_heads: int  # num_query_groups if GQA
    head_dim: int

    inter_sz: int
    glu: bool
    moe_cfg: Optional[MoeCfg]

    rotary_embed: bool

    dropout: bool

    vocab_sz: int
    tie_embeddings: bool

    act_ckpting_type: ActivationCheckpointingType

    bucket_size_bytes: int

    # TODO. assuming mixed precision here.
    bits_per_parameter: int = 16
    bits_per_grad: int = 32
    bits_per_optim_state: int = 32

    def __post_init__(self) -> None:
        if self.n_layers % (self.parallelism_cfg.pp * self.parallelism_cfg.vpp) != 0:
            raise ValueError(
                f"number of layers {self.n_layers} is not divisible by the product "
                f"of PP={self.parallelism_cfg.pp} and VPP={self.parallelism_cfg.vpp}"
            )

        if self.moe_cfg:
            if not (self.moe_cfg.moe_frequency * self.n_layers).is_integer():
                raise ValueError(
                    f"invalid moe frequency {self.moe_cfg.moe_frequency} for layer number {self.n_layers}"
                )
            self.n_moe_layers = int(self.moe_cfg.moe_frequency * self.n_layers)
            self.n_nml_layers = self.n_layers - self.n_moe_layers
        else:
            self.n_moe_layers = 0
            self.n_nml_layers = self.n_layers

        n_experts = self.moe_cfg.n_experts if self.moe_cfg else 1

        self.embed_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz, self.vocab_sz),
            partition_spec={1: self.parallelism_cfg.tp},  # vocab-parallel
            bits_per_elt=self.bits_per_parameter,
        )
        self.pre_attn_norm_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz,),
            partition_spec={},  # replicated
            bits_per_elt=self.bits_per_parameter,
        )
        self.qkv_weight = TensorRepr(
            unpartitioned_shape=(
                self.hidden_sz,
                # following common practice of merging Q + K + V matmuls.
                (self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim,
            ),
            partition_spec={1: self.parallelism_cfg.tp},  # col parallel
            bits_per_elt=self.bits_per_parameter,
        )
        self.attn_out_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz, self.hidden_sz),
            partition_spec={0: self.parallelism_cfg.tp},  # row parallel
            bits_per_elt=self.bits_per_parameter,
        )
        self.router_weight = TensorRepr(
            unpartitioned_shape=(
                self.hidden_sz,
                self.moe_cfg.n_experts if self.moe_cfg else 0,
            ),
            partition_spec={},
            bits_per_elt=self.bits_per_parameter,
        )
        self.mlp_up_weight = TensorRepr(
            unpartitioned_shape=(
                self.hidden_sz,
                # following common practice of merging up + gate matmuls in the event
                # we're using GLU.
                (self.inter_sz * 2) if self.glu else self.inter_sz,
            ),
            partition_spec={1: self.parallelism_cfg.tp},  # col parallel
            bits_per_elt=self.bits_per_parameter,
        )
        self.mlp_down_weight = TensorRepr(
            unpartitioned_shape=(self.inter_sz, self.hidden_sz),
            partition_spec={0: self.parallelism_cfg.tp},  # row parallel
            bits_per_elt=self.bits_per_parameter,
        )

        if self.moe_cfg is not None:
            assert self.parallelism_cfg.expert_mesh is not None
            self.mlp_up_exp_weight = (
                TensorRepr(
                    # following common practice of merging up + gate matmuls in the event
                    # we're using GLU.
                    unpartitioned_shape=(
                        n_experts,
                        self.hidden_sz,
                        (self.moe_cfg.expert_inter_sz * 2)
                        if self.glu
                        else self.moe_cfg.expert_inter_sz,
                    ),
                    partition_spec={
                        0: self.parallelism_cfg.expert_mesh.ep,
                        2: self.parallelism_cfg.expert_mesh.tp,
                    },  # col parallel
                    bits_per_elt=self.bits_per_parameter,
                )
                if self.moe_cfg is not None
                else None
            )
            self.mlp_down_exp_weight = (
                TensorRepr(
                    unpartitioned_shape=(
                        n_experts,
                        self.moe_cfg.expert_inter_sz,
                        self.hidden_sz,
                    ),
                    partition_spec={
                        0: self.parallelism_cfg.expert_mesh.ep,
                        1: self.parallelism_cfg.expert_mesh.tp,
                    },  # row parallel
                    bits_per_elt=self.bits_per_parameter,
                )
                if self.moe_cfg is not None
                else None
            )
        else:
            self.mlp_up_exp_weight = None
            self.mlp_down_exp_weight = None

        self.pre_mlp_norm_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz,),
            partition_spec={},  # replicated
            bits_per_elt=self.bits_per_parameter,
        )

        if self.parallelism_cfg.zero_level != ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER:
            raise NotImplementedError

        self.states = ModelStates(
            params_shard=TensorRepr(
                unpartitioned_shape=(
                    self.__get_n_total_params(
                        spmd_partitioned=True,
                        mpmd_partitioned=True,
                    ),
                ),
                partition_spec={},
                bits_per_elt=self.bits_per_parameter,
                enforce_evenly_partitionable=False,
            ),
            opt_states=DistributedAdamOptimizerStates(
                n_params=self.__get_n_total_params(
                    spmd_partitioned=True,
                    mpmd_partitioned=True,
                ),
                store_param_remainders=True,  # TODO. should be configurable
                dp=self.parallelism_cfg.dp,
            ),
        )

    def get_single_microbatch_fwd_flops(self) -> float:
        return (
            2  # FLOPs/MAC
            * 1  # factor for forward only (1 GEMMs per op)
            * self.microbatch_sz
            * self.sequence_len
            * self.__get_n_active_params(partitioned=False)
        )

    def get_single_microbatch_bwd_flops(self) -> float:
        return (
            2  # FLOPs/MAC
            * 2  # factor for backward only (2 GEMMs per op)
            * self.microbatch_sz
            * self.sequence_len
            * self.__get_n_active_params(partitioned=False)
        )

    def get_n_total_params(self, partitioned: bool) -> int:
        return self.__get_n_total_params(
            spmd_partitioned=partitioned,
            mpmd_partitioned=partitioned,
        )

    def get_n_active_params(self, partitioned: bool) -> int:
        return self.__get_n_active_params(partitioned=partitioned)

    def activation_size_per_microbatch_per_layer(self) -> Size:
        return Size(
            numel=self.__dense_activation_numel_per_microbatch_per_layer(),
            bits_per_element=self.bits_per_parameter,
        )

    def vpp_penalty(self) -> float:
        """interleaved schedule requires storing activations for (1 + (p - 1)/pm)
        more layers."""
        if self.parallelism_cfg.vpp == 1:
            return 1.0

        return 1 + (self.parallelism_cfg.pp - 1) / (
            self.parallelism_cfg.pp * self.parallelism_cfg.vpp
        )

    def layers_per_pp_stage(self) -> int:
        return sum(self.__n_layers(mpmd_partitioned=True, moe=moe) for moe in [False, True])

    def grad_bucket_numel(self) -> int:
        return ceil_divide(self.bucket_size_bytes, self.bits_per_grad // 8)

    def __get_n_total_params(self, spmd_partitioned: bool, mpmd_partitioned: bool) -> int:
        return _sum(
            # we'll give the number of parameters on the most heavily loaded pipeline stage
            # if PP=1 then the only pipeline stage must store both embedding and LM head.
            (1 if (mpmd_partitioned and self.parallelism_cfg.pp > 1) or self.tie_embeddings else 2)
            * self.__get_embedding_or_lm_head_size(spmd_partitioned=spmd_partitioned),
            # add in the transformer blocks
            sum(
                self.__n_layers(mpmd_partitioned=mpmd_partitioned, moe=moe)
                * self.__get_transformer_block_n_params(
                    spmd_partitioned=spmd_partitioned,
                    moe=moe,
                    active=False,
                    experts_per_token=None,
                )
                # TODO. clean this up
                for moe in ([False, True] if self.moe_cfg is not None else [False])
            ),
        )

    def __get_n_active_params(self, partitioned: bool) -> int:
        if not self.moe_cfg:
            return self.__get_n_total_params(
                spmd_partitioned=partitioned,
                mpmd_partitioned=partitioned,
            )

        # we'll give the number of parameters on the most heavily loaded pipeline stage
        # if PP=1 then the only pipeline stage must store both embedding and LM head.
        return _sum(
            # embedding/lmhead
            (1 if (partitioned and self.parallelism_cfg.pp > 1) or self.tie_embeddings else 2)
            * self.__get_embedding_or_lm_head_size(spmd_partitioned=partitioned),
            # transformer blocks
            sum(
                self.__n_layers(mpmd_partitioned=partitioned, moe=moe)
                * self.__get_transformer_block_n_params(
                    spmd_partitioned=partitioned,
                    moe=moe,
                    active=True,
                    experts_per_token=self.moe_cfg.experts_per_token,
                )
                for moe in [False, True]
            ),
        )

    def __get_embedding_or_lm_head_size(self, spmd_partitioned: bool) -> int:
        return self.embed_weight.size(partitioned=spmd_partitioned).numel()

    def __get_transformer_block_n_params(
        self,
        spmd_partitioned: bool,
        moe: bool,
        # TODO. would rather not expose these.
        active: bool,
        experts_per_token: Optional[int],
    ) -> int:
        assert active == (experts_per_token is not None)

        if moe:
            assert self.moe_cfg is not None
            assert self.mlp_up_exp_weight is not None
            assert self.mlp_down_exp_weight is not None
            mlp_params = (
                # if we're trying to compute active params, then we account for the
                # fact that we'll apply topk mlps per token.
                experts_per_token  # type: ignore[operator]
                * _sum(
                    self.mlp_up_exp_weight.numel(partitioned=spmd_partitioned)
                    // self.moe_cfg.n_experts,
                    self.mlp_down_exp_weight.numel(partitioned=spmd_partitioned)
                    // self.moe_cfg.n_experts,
                )
                if active
                else _sum(
                    self.mlp_up_exp_weight.numel(partitioned=spmd_partitioned),
                    self.mlp_down_exp_weight.numel(partitioned=spmd_partitioned),
                )
            )
        else:
            mlp_params = _sum(
                self.mlp_up_weight.numel(partitioned=spmd_partitioned),
                self.mlp_down_weight.numel(partitioned=spmd_partitioned),
            )

        return _sum(
            self.pre_attn_norm_weight.numel(partitioned=spmd_partitioned),
            self.qkv_weight.numel(partitioned=spmd_partitioned),
            self.attn_out_weight.numel(partitioned=spmd_partitioned),
            self.pre_mlp_norm_weight.numel(partitioned=spmd_partitioned),
            mlp_params,
        )

    def __n_layers(self, mpmd_partitioned: bool, moe: bool) -> int:
        # TODO. we're making the assumption that MoE and non-MoE layers
        # can be evenly partitioned.
        n_layers = self.n_moe_layers if moe else self.n_nml_layers
        if mpmd_partitioned:
            n_layers = safe_divide(n_layers, self.parallelism_cfg.pp)

        return n_layers

    # TODO. probably needs to be adjusted for MoE
    def __dense_activation_numel_per_microbatch_per_layer(self) -> int:
        """
        See: Reducing Activation Recomputation in Large Transformer Models
        https://arxiv.org/pdf/2205.05198.pdf
        """
        sbh = self.sequence_len * self.microbatch_sz * self.hidden_sz
        sbq = self.sequence_len * self.microbatch_sz * self.n_q_heads * self.head_dim
        sbk = self.sequence_len * self.microbatch_sz * self.n_kv_heads * self.head_dim
        sbv = self.sequence_len * self.microbatch_sz * self.n_kv_heads * self.head_dim

        n_local_mlps = (
            1
            if self.moe_cfg is None
            else safe_divide(self.moe_cfg.n_experts, self.parallelism_cfg.expert_mesh.ep)
        )
        sbi = (
            self.sequence_len * self.microbatch_sz * self.inter_sz
            if self.moe_cfg is None
            else self.expert_capacity() * self.moe_cfg.expert_inter_sz
        )

        if self.act_ckpting_type == ActivationCheckpointingType.FULL:
            return self.__sp_partition_if_on(sbh)  # just the block input
        elif self.act_ckpting_type == ActivationCheckpointingType.SUPER_SELECTIVE:
            return _sum(
                # LAYERNORM 1
                # - output is recomputed
                # QKV (col parallel linear)
                self.__tp_partition(sbq),  # Q - attn input
                self.__tp_partition(sbk),  # K - attn input
                self.__tp_partition(sbv),  # V - attn input
                # ROTARY EMBEDDINGS
                self.__tp_partition(sbq) if self.rotary_embed else 0,  # Q rotary
                self.__tp_partition(sbk) if self.rotary_embed else 0,  # K rotary
                # SELF ATTENTION
                # - skipping intermediates (checkpointed by FlashAttention)
                self.__tp_partition(sbh),  # attn output
                # DOWN PROJ
                # - output deallocated (dropout doesn't need to store)
                # DROPOUT
                # - dropout mask recomputed
                # - deallocated - residual doesn't need to store
                # RESIDUAL
                self.__sp_partition_if_on(sbh),  # stored by norm2, residual2
                # LAYERNORM 2
                self.__sp_partition_if_on(sbh),
                # TOKEN PERMUTATION (ALLGATHERED)
                # TODO.
                # output is recomputed
                # MLP UP/GATE (col parallel linear)
                n_local_mlps * self.__tp_partition(2 * sbi if self.glu else sbi),  # SwiGLU input
                # SwiGLU
                # TODO. swiglu output
                # MLP DOWN (row parallel linear)
                # - output deallocated - dropout doesn't need to store
                # TODO. all-to-all token unpermutation
                # TOKEN UNPERMUTATION
                # - output is recomputed
                # DROPOUT
                # - dropout mask recomputed
                # - output deallocated - residual doesn't need to store
                # RESIDUAL
                self.__sp_partition_if_on(sbh),  # needed by next norm1, residual1
            )
        elif self.act_ckpting_type == ActivationCheckpointingType.SELECTIVE:
            return _sum(
                # LAYERNORM 1
                self.__sp_partition_if_on(sbh),  # output - QKV input
                # QKV PROJ (col parallel linear)
                self.__tp_partition(sbq),  # Q - attn input
                self.__tp_partition(sbk),  # K - attn input
                self.__tp_partition(sbv),  # V - attn input
                # ROTARY EMBEDDINGS
                self.__tp_partition(sbq) if self.rotary_embed else 0,  # Q rotary
                self.__tp_partition(sbk) if self.rotary_embed else 0,  # K rotary
                # SELF ATTENTION
                # - skipping intermediates (checkpointed by FlashAttention)
                self.__tp_partition(sbh),  # needed by down proj
                # DOWN PROJ
                # - output deallocated (dropout doesn't need to store)
                # DROPOUT
                self.__sp_partition_if_on(int(0.5 * sbh)) if self.dropout else 0,  # dropout mask
                # -  output deallocated: residual doesn't need to store
                # RESIDUAL
                self.__sp_partition_if_on(sbh),  # needed by norm2, resid2
                # LAYERNORM 2
                self.__sp_partition_if_on(sbh),  # up/gate input
                # MLP UP/GATE (col parallel linear)
                n_local_mlps * self.__tp_partition(2 * sbi if self.glu else sbi),  # SwiGLU input
                # SwiGLU
                self.__tp_partition(sbi),  # SiLU output
                self.__tp_partition(sbi),  # gate output
                # MLP DOWN (row parallel linear)
                # - output deallocated - dropout doesn't need to store
                # DROPOUT
                self.__sp_partition_if_on(int(0.5 * sbh)) if self.dropout else 0,  # dropout mask
                # - output deallocated - residual doesn't need to store
                # RESIDUAL
                # input to next norm1, resid1
                self.__sp_partition_if_on(sbh),
            )
        elif self.act_ckpting_type == ActivationCheckpointingType.NONE:
            raise NotImplementedError(
                "not yet implemented. selective will give a good approximation "
                "if you use flash attention"
            )
        else:
            raise ValueError(f"unhandled checkpointing_type={self.act_ckpting_type}")

    def __tp_partition(self, unpartitoned_numel: int) -> int:
        x = unpartitoned_numel
        for parallelism_degree in [self.parallelism_cfg.cp, self.parallelism_cfg.tp]:
            x = safe_divide(x, parallelism_degree)

        return x

    def __sp_partition_if_on(self, unpartitioned_numel: int) -> int:
        parallelism_degrees = [self.parallelism_cfg.cp]
        if self.parallelism_cfg.sp_enabled:
            parallelism_degrees.append(self.parallelism_cfg.tp)

        x = unpartitioned_numel
        for parallelism_degree in parallelism_degrees:
            x = safe_divide(x, parallelism_degree)

        return x

    def expert_capacity(self) -> int:
        """Returns the number of tokens that can be processed by each expert."""
        if self.moe_cfg is None:
            raise RuntimeError
        assert self.parallelism_cfg.expert_mesh is not None

        return safe_divide(
            int(
                safe_divide(self.sequence_len, self.parallelism_cfg.tp * self.parallelism_cfg.cp)
                * self.microbatch_sz
                * self.moe_cfg.experts_per_token
                * self.parallelism_cfg.expert_mesh.ep
                * self.moe_cfg.capacity_factor
            ),
            self.moe_cfg.n_experts,
        )
