import dataclasses
from enum import Enum

from .configurations import ActivationCheckpointingType
from .data import Size, TensorRepr
from .math import safe_divide


@dataclasses.dataclass
class ParallelConfig:
    class ZeroLevel(int, Enum):
        NONE = 0
        PARTITION_OPTIMIZER = 1
        PARTITION_GRADIENTS = 2
        PARTITION_PARAMETERS = 3

    tp: int  # tensor parallel degree
    pp: int  # pipeline parallel degree
    dp: int  # data parallel degree

    vpp: int  # virtual pipeline parallel degree

    sp_enabled: bool  # sequence parallel

    zero_level: ZeroLevel

    def mp_degree(self) -> int:
        return self.tp * self.pp

    def world_size(self) -> int:
        return self.tp * self.pp * self.dp


class DistributedAdamOptimizerStates:
    """Optimizer states from Apex DistributedFusedAdam.
    see: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md
    and: https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/distributed_fused_adam.py

    NOTE: shards will be larger in reality due to alignment requirements and
    unfilled buckets.
    """

    def __init__(self, n_params: int, store_param_remainders: bool, dp: int) -> None:
        self.param_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_degree=dp,
            # apex has a optimization to avoid storing information that's redundant
            # between fp32 and fp16 weights. it can instead store an extra 16
            # bits of precision, which can be combined with bf16 weights to yield
            # fp32 weights
            bits_per_elt=16 if store_param_remainders else 32,
            enforce_evenly_partitionable=False,
        )
        self.exp_avg_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_degree=dp,
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )
        self.exp_avg_sq_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_degree=dp,
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
            partition_degree=1,
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
                f"TOTAL           : {(params_bytes + opt_states_bytes) / (1024 ** 3):.2f}GiB",
            ]
        )


def _sum(*summands):
    return sum(list(summands))


@dataclasses.dataclass
class ThreeDParallelModel:
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

    rotary_embed: bool

    vocab_sz: int
    tie_embeddings: bool

    act_ckpting_type: ActivationCheckpointingType

    # TODO. assuming mixed precision here.
    bits_per_parameter: int = 16
    bits_per_grad: int = 32
    bits_per_optim_state: int = 32

    def __post_init__(self) -> None:
        self.embed_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz, self.vocab_sz),
            partition_degree=self.parallelism_cfg.tp,
            bits_per_elt=self.bits_per_parameter,
        )
        self.norm1_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz,),
            partition_degree=1,
            bits_per_elt=self.bits_per_parameter,
        )
        self.qkv_weight = TensorRepr(
            unpartitioned_shape=(
                self.hidden_sz,
                (self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim,
            ),
            partition_degree=self.parallelism_cfg.tp,
            bits_per_elt=self.bits_per_parameter,
        )
        self.attn_out_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz, self.hidden_sz),
            partition_degree=self.parallelism_cfg.tp,
            bits_per_elt=self.bits_per_parameter,
        )
        self.mlp_up_weight = TensorRepr(
            # following common practice of merging up + gate matmuls in the event
            # we're using GLU.
            unpartitioned_shape=(
                self.hidden_sz,
                (self.inter_sz * 2) if self.glu else self.inter_sz,
            ),
            partition_degree=self.parallelism_cfg.tp,
            bits_per_elt=self.bits_per_parameter,
        )
        self.mlp_down_weight = TensorRepr(
            unpartitioned_shape=(self.inter_sz, self.hidden_sz),
            partition_degree=self.parallelism_cfg.tp,
            bits_per_elt=self.bits_per_parameter,
        )
        self.norm2_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz,),
            partition_degree=1,
            bits_per_elt=self.bits_per_parameter,
        )

        if self.n_layers % (self.parallelism_cfg.pp * self.parallelism_cfg.vpp) != 0:
            raise ValueError(
                f"number of layers {self.n_layers} is not divisible by the product "
                f"of PP={self.parallelism_cfg.pp} and VPP={self.parallelism_cfg.vpp}"
            )

    def get_total_n_params(self, partitioned: bool) -> int:
        embedding_or_lm_head_n_params = self.__get_embedding_or_lm_head_size(
            partitioned=partitioned
        )
        transformer_block_n_params = self.__get_transformer_block_n_params(partitioned=partitioned)

        if partitioned:
            # we'll give the number of parameters on the most heavily loaded pipeline stage
            return (
                embedding_or_lm_head_n_params
                + safe_divide(self.n_layers, self.parallelism_cfg.pp) * transformer_block_n_params
            )
        else:
            embedding_or_lm_head_factor = 1 if self.tie_embeddings else 2
            return (
                embedding_or_lm_head_factor * embedding_or_lm_head_n_params
                + self.n_layers * transformer_block_n_params
            )

    def get_partitioned_states(self, training: bool) -> ModelStates:
        if not training:  # TODO. cleanup
            raise NotImplementedError

        n_params_most_loaded_pp_stage = self.__get_embedding_or_lm_head_size(
            partitioned=True
        ) + self.layers_per_pp_stage() * self.__get_transformer_block_n_params(partitioned=True)

        if self.parallelism_cfg.zero_level != ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER:
            raise NotImplementedError

        return ModelStates(
            params_shard=TensorRepr(
                unpartitioned_shape=(self.get_total_n_params(partitioned=False),),
                partition_degree=self.parallelism_cfg.mp_degree(),
                bits_per_elt=self.bits_per_parameter,
                enforce_evenly_partitionable=False,
            ),
            opt_states=DistributedAdamOptimizerStates(
                n_params=n_params_most_loaded_pp_stage,
                store_param_remainders=True,  # TODO. should be configurable
                dp=self.parallelism_cfg.dp,
            ),
        )

    def __get_embedding_or_lm_head_size(self, partitioned: bool) -> int:
        return self.embed_weight.size(partitioned=partitioned).numel()

    def __get_transformer_block_n_params(self, partitioned: bool) -> int:
        return _sum(
            self.norm1_weight.size(partitioned=partitioned).numel(),
            self.qkv_weight.size(partitioned=partitioned).numel(),
            self.attn_out_weight.size(partitioned=partitioned).numel(),
            self.norm2_weight.size(partitioned=partitioned).numel(),
            self.mlp_up_weight.size(partitioned=partitioned).numel(),
            self.mlp_down_weight.size(partitioned=partitioned).numel(),
        )

    def activation_size_per_microbatch_per_layer(self) -> Size:
        return Size(
            numel=self.__activation_numel_per_microbatch_per_layer(),
            bits_per_element=self.bits_per_parameter,
        )

    def max_inflight_microbatches(self) -> int:
        return self.parallelism_cfg.pp

    def vpp_penalty(self) -> int:
        """interleaved schedule requires storing activations for (1 + (p - 1)/pm)
        more layers."""
        if self.parallelism_cfg.vpp == 1:
            return 1.0

        return 1 + (self.parallelism_cfg.pp - 1) / (
            self.parallelism_cfg.pp * self.parallelism_cfg.vpp
        )

    def layers_per_pp_stage(self) -> int:
        return safe_divide(self.n_layers, self.parallelism_cfg.pp)

    def __activation_numel_per_microbatch_per_layer(self) -> int:
        """
        See: Reducing Activation Recomputation in Large Transformer Models
        https://arxiv.org/pdf/2205.05198.pdf
        """
        sbh = self.sequence_len * self.microbatch_sz * self.hidden_sz
        sbi = self.sequence_len * self.microbatch_sz * self.inter_sz
        sbq = self.sequence_len * self.microbatch_sz * self.n_q_heads * self.head_dim
        sbk = self.sequence_len * self.microbatch_sz * self.n_kv_heads * self.head_dim
        sbv = self.sequence_len * self.microbatch_sz * self.n_kv_heads * self.head_dim

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
                self.__tp_partition(sbq if self.rotary_embed else 0),  # Q rotary
                self.__tp_partition(sbk if self.rotary_embed else 0),  # K rotary
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
                # output is recomputed
                # MLP UP/GATE (col parallel linear)
                self.__tp_partition(2 * sbi if self.glu else sbi),  # SwiGLU input
                # SwiGLU
                # - output is recomputed
                # MLP DOWN (row parallel linear)
                # - output deallocated - dropout doesn't need to store
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
                self.__tp_partition(sbq if self.rotary_embed else 0),  # Q rotary
                self.__tp_partition(sbk if self.rotary_embed else 0),  # K rotary
                # SELF ATTENTION
                # - skipping intermediates (checkpointed by FlashAttention)
                self.__tp_partition(sbh),  # needed by down proj
                # DOWN PROJ
                # - output deallocated (dropout doesn't need to store)
                # DROPOUT
                self.__sp_partition_if_on(0.5 * sbh),  # dropout mask
                # -  output deallocated: residual doesn't need to store
                # RESIDUAL
                self.__sp_partition_if_on(sbh),  # needed by norm2, resid2
                # LAYERNORM 2
                self.__sp_partition_if_on(sbh),  # up/gate input
                # MLP UP/GATE (col parallel linear)
                self.__tp_partition(2 * sbi if self.glu else sbi),  # SwiGLU input
                # SwiGLU
                self.__tp_partition(sbi),  # SiLU output
                self.__tp_partition(sbi),  # gate output
                # MLP DOWN (row parallel linear)
                # - output deallocated - dropout doesn't need to store
                # DROPOUT
                self.__sp_partition_if_on(0.5 * sbh),  # dropout mask
                # - output deallocated - residual doesn't need to store
                # RESIDUAL
                # input to next norm1, resid1
                self.__sp_partition_if_on(sbh),
            )
        elif self.act_ckpting_type == ActivationCheckpointingType.NONE:
            raise NotImplementedError("not yet implemented")
            return _sum(
                # LAYERNORM 1
                sbh,  # layernorm 1 input
                # ATTENTION
                sbh,  # QKV matmul input
                # skipping Q @ K.T (checkpointed by FlashAttention)
                # skipping Softmax (checkpointed by FlashAttention)
                # skipping Softmax dropout (checkpointed by FlashAttention)
                sbh,  # down proj input.
                # LAYERNORM 2
                sbh,  # layernorm 2 input
                # MLP
                sbh,  # up/gate input
                sbi,  # activation input
                sbi,  # down input TODO. pass inter_sz
                0.5 * sbh,  # dropout mask
                # TODO. this is sort of hacky, dropout mask is 1 byte so setting as half numel
            )
        else:
            raise ValueError(f"unhandled checkpointing_type={self.act_ckpting_type}")

    def __tp_partition(self, unpartitoned_numel: int) -> int:
        return safe_divide(unpartitoned_numel, self.parallelism_cfg.tp)

    def __sp_partition_if_on(self, unpartitioned_numel: int) -> int:
        if not self.parallelism_cfg.sp_enabled:
            return unpartitioned_numel
        return safe_divide(unpartitioned_numel, self.parallelism_cfg.tp)
