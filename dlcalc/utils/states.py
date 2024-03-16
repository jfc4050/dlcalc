import dataclasses
from enum import Enum

from .configurations import ActivationCheckpointingType
from .math import product, safe_divide
from .size import Size, TensorRepr


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
    """
    see: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md
    """

    def __init__(self, n_params: int, store_param_remainders: bool, dp: int) -> None:
        self.param_shard = TensorRepr(
            shape=(n_params,),
            partition_degree=dp,
            # storing param remainders means that the optimizer stores the extra
            # 16 bits
            bits_per_elt=16 if store_param_remainders else 32,
            enforce_evenly_partitionable=False,
        )
        self.exp_avg_shard = TensorRepr(
            shape=(n_params,),
            partition_degree=dp,
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )
        self.exp_avg_sq_shard = TensorRepr(
            shape=(n_params,),
            partition_degree=dp,
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )


class States:
    def __init__(self, params: TensorRepr, grads: TensorRepr, optim_states: TensorRepr) -> None:
        self.params = params
        self.grads = grads
        self.optim_states = optim_states

    def total_bytes_partitioned(self) -> int:
        return sum(
            [
                self.params.size(partitioned=True).bytes(),
                self.grads.size(partitioned=True).bytes() if self.grads is not None else 0,
                self.optim_states.size(partitioned=True).bytes()
                if self.optim_states is not None
                else 0,
            ]
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"params       : {self.params.size(partitioned=True)}",
                f"grads        : {self.grads.size(partitioned=True)}",
                f"optim_states : {self.optim_states.size(partitioned=True)}",
                f"TOTAL        : {self.total_bytes_partitioned() / (1024 ** 3):.2f}GiB",
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

    def get_total_n_params_unpartitioned(self) -> int:
        return 2 * self.__get_embedding_or_lm_head_size(
            partitioned=False
        ) + self.n_layers * self.__get_transformer_block_n_params(partitioned=False)

    def get_partitioned_states(self, training: bool) -> States:
        n_params_most_loaded_pp_stage = self.__get_embedding_or_lm_head_size(
            partitioned=True
        ) + self.layers_per_pp_stage() * self.__get_transformer_block_n_params(partitioned=True)

        if self.parallelism_cfg.zero_level != ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER:
            raise NotImplementedError

        return States(
            params=TensorRepr(
                unpartitioned_shape=(n_params_most_loaded_pp_stage,),
                partition_degree=self.parallelism_cfg.dp,
                bits_per_elt=self.bits_per_parameter,
                enforce_evenly_partitionable=False,
            ),
            # for gradient and optimizer partitioning, see
            # https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/distributed_fused_adam.py
            # https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md
            grads=TensorRepr(
                unpartitioned_shape=(n_params_most_loaded_pp_stage,),
                partition_degree=1,
                bits_per_elt=self.bits_per_grad,
            )
            if training
            else None,
            optim_states=TensorRepr(
                unpartitioned_shape=(3 * n_params_most_loaded_pp_stage,),
                partition_degree=self.parallelism_cfg.dp,
                bits_per_elt=self.bits_per_optim_state,
                enforce_evenly_partitionable=False,
            )
            if training
            else None,
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
        sbkv = self.sequence_len * self.microbatch_sz * self.n_kv_heads * self.head_dim

        if self.act_ckpting_type == ActivationCheckpointingType.FULL:
            return self.__sp_partition_if_on(sbh)  # just the block input
        elif self.act_ckpting_type == ActivationCheckpointingType.SUPER_SELECTIVE:
            return _sum(
                # LAYERNORM 1
                # - output is recomputed
                # QKV (col parallel linear)
                self.__tp_partition(sbq),  # Q - attn input
                self.__tp_partition(sbkv),  # K - attn input
                self.__tp_partition(sbkv),  # V - attn input
                # ROTARY EMBEDDINGS
                self.__tp_partition(sbq if self.rotary_embed else 0),  # Q rotary
                self.__tp_partition(sbkv if self.rotary_embed else 0),  # K rotary
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
                self.__tp_partition(sbkv),  # K - attn input
                self.__tp_partition(sbkv),  # V - attn input
                # ROTARY EMBEDDINGS
                self.__tp_partition(sbq if self.rotary_embed else 0),  # Q rotary
                self.__tp_partition(sbkv if self.rotary_embed else 0),  # K rotary
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
