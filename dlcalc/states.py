import dataclasses
from enum import Enum

from dlcalc.utils.math import safe_divide
from .utils.size import Size
from .utils.configurations import ActivationCheckpointingType


@dataclasses.dataclass
class ParallelismConfig:
    class ZeroLevel(int, Enum):
        NONE = 0
        PARTITION_OPTIMIZER = 1
        PARTITION_GRADIENTS = 2
        PARTITION_PARAMETERS = 3

    tp: int
    pp: int
    dp: int

    # virtual pipeline parallel degree
    vpp: int

    sp_enabled: bool

    zero_level: ZeroLevel

    def mp_degree(self) -> int:
        return self.tp * self.pp

    def world_size(self) -> int:
        return self.tp * self.pp * self.dp


@dataclasses.dataclass
class Zero3Model:
    n_params: int
    world_size: int

    # TODO. assuming mixed precision here.
    bytes_per_parameter: int = 2

    def params_per_rank(self):
        return self.n_params / self.world_size


class States:
    def __init__(self, params: Size, grads: Size, optim_states: Size) -> None:
        self.params = params
        self.grads = grads
        self.optim_states = optim_states

    def total_bytes(self) -> int:
        return sum(
            [
                self.params.bytes(),
                self.grads.bytes() if self.grads is not None else 0,
                self.optim_states.bytes() if self.optim_states is not None else 0,
            ]
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"params       : {self.params}",
                f"grads        : {self.grads}",
                f"optim_states : {self.optim_states}",
                f"TOTAL        : {self.total_bytes() / (1024 ** 3):.2f}GiB",
            ]
        )


def _sum(*summands):
    return sum(list(summands))


@dataclasses.dataclass
class ThreeDParallelModel:
    """
    a great blog post summarizing many of the equations coded below:
    https://eleutherai.notion.site/Transformers-Math-101-d2fcfc7a25d446388fde97821ad2412a

    also see:
    https://gist.github.com/Quentin-Anthony/f43939791a7ceb0b01a4937308317be5
    """

    parallelism_cfg: ParallelismConfig

    sequence_len: int
    microbatch_sz: int

    hidden_sz: int

    n_layers: int

    n_q_heads: int
    n_kv_heads: int  # num_query_groups if GQA
    head_dim: int

    inter_sz: int
    glu: bool

    vocab_sz: int

    act_ckpting_type: ActivationCheckpointingType

    # TODO. assuming mixed precision here.
    bytes_per_parameter: int = 2

    def get_transformer_block_n_params(self) -> int:
        numel = _sum(
            # norm1,
            self.hidden_sz,
            # qkv_proj (col parallel)
            self.hidden_sz * ((self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim),
            # attn out_proj (row parallel)
            self.hidden_sz * self.hidden_sz,
            # norm2
            self.hidden_sz,
            # MLP layer 1 (col parallel)
            self.hidden_sz * ((self.inter_sz * 2) if self.glu else self.inter_sz),
            # MLP layer 2 (row parallel)
            self.inter_sz * self.hidden_sz,
        )

        return Size(numel=numel, bytes_per_element=self.bytes_per_parameter)

    def get_embedding_or_lm_head_n_params(self) -> int:
        return Size(
            numel=self.hidden_sz * self.vocab_sz / self.parallelism_cfg.tp,
            bytes_per_element=self.bytes_per_parameter,
        )

    def get_states(self, training: bool) -> States:
        params_in_most_loaded_pp_stage = (
            self.get_embedding_or_lm_head_n_params()
            + self.layers_per_pp_stage() * self.get_transformer_block_n_params()
        ).numel()
        tp_params_most_loaded_pp_stage = (
            params_in_most_loaded_pp_stage / self.parallelism_cfg.tp
        )

        return States(
            params=Size(
                numel=tp_params_most_loaded_pp_stage,
                bytes_per_element=self.bytes_per_parameter,
            ),
            grads=Size(
                numel=tp_params_most_loaded_pp_stage,
                bytes_per_element=self.bytes_per_parameter,
            )
            if training
            else None,
            # see:
            # https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md
            optim_states=Size(
                numel=(
                    # fp32 params
                    tp_params_most_loaded_pp_stage
                    +
                    # momentum/variance
                    2 * tp_params_most_loaded_pp_stage / self.parallelism_cfg.dp
                ),
                # TODO. assuming AMP here.
                bytes_per_element=4,
            )
            if training
            else None,
        )

    def activation_size_per_microbatch_per_layer(self) -> Size:
        # TODO. assuming half precision here.
        return Size(
            numel=self.__activation_numel_per_microbatch_per_layer(),
            bytes_per_element=self.bytes_per_parameter,
        )

    def max_inflight_microbatches(self) -> int:
        if self.parallelism_cfg.vpp > 1:
            # TODO. need to understand this VPP penalty better
            interleaved_schedule_mem_penalty = 1 + (self.parallelism_cfg.pp - 1) / (
                self.parallelism_cfg.pp * self.parallelism_cfg.vpp
            )
            return interleaved_schedule_mem_penalty * self.parallelism_cfg.pp
        else:
            return self.parallelism_cfg.pp

    def kv_cache_size(self, gen_batch_sz: int) -> Size:
        return Size(
            numel=self.n_layers
            * 2  # one for each of K and V
            * gen_batch_sz
            * self.sequence_len
            * self.n_kv_heads  # may be different than num heads if GQA, MQA.
            * self.head_dim,
            bytes_per_element=self.bytes_per_parameter,
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
            return self.__sp_partition_if_applicable(sbh)  # just the block input
        elif self.act_ckpting_type == ActivationCheckpointingType.SUPER_SELECTIVE:
            return _sum(
                # LAYERNORM 1
                # - output is recomputed
                # QKV (col parallel linear)
                self.__tp_partition(sbq),  # Q - attn input
                self.__tp_partition(sbkv),  # K - attn input
                self.__tp_partition(sbkv),  # V - attn input
                # SELF ATTENTION
                # - skipping Q @ K.T (checkpointed by FlashAttention)
                # - skipping Softmax (checkpointed by FlashAttention)
                # - skipping Softmax dropout (checkpointed by FlashAttention)
                self.__tp_partition(sbh),  # attn output
                # DOWN PROJ
                # - deallocated (dropout doesn't need to store)
                # DROPOUT
                # - dropout mask recomputed
                # - deallocated - residual doesn't need to store
                # RESIDUAL
                self.__sp_partition_if_applicable(sbh),  # stored by norm2, residual2
                # LAYERNORM 2
                # output is recomputed
                # MLP UP/GATE (col parallel linear)
                self.__tp_partition(2 * sbi if self.glu else sbi),  # SwiGLU input
                # SwiGLU
                # - output is recomputed
                # MLP DOWN (row parallel linear)
                self.__sp_partition_if_applicable(sbh),  # TODO. why not deallocated?
                # DROPOUT
                # - dropout mask recomputed
                # - output deallocated - residual doesn't need to store
                # RESIDUAL
                self.__sp_partition_if_applicable(sbh),  # for by next norm1, residual1
            )
        elif self.act_ckpting_type == ActivationCheckpointingType.SELECTIVE:
            return _sum(
                # LAYERNORM 1
                self.__sp_partition_if_applicable(sbh),  # output - QKV input
                # QKV PROJ (col parallel linear)
                self.__tp_partition(sbq),  # Q - attn input
                self.__tp_partition(sbkv),  # K - attn input
                self.__tp_partition(sbkv),  # V - attn input
                # SELF ATTENTION
                # skipping Q @ K.T (checkpointed by FlashAttention)
                # skipping Softmax (checkpointed by FlashAttention)
                # skipping Softmax dropout (checkpointed by FlashAttention)
                self.__tp_partition(sbh),  # needed by down proj
                # DOWN PROJ
                # - deallocated (dropout doesn't need to store)
                # DROPOUT
                self.__sp_partition_if_applicable(0.5 * sbh),  # dropout mask
                # -  output deallocated: residual doesn't need to store
                # RESIDUAL
                # needed by layernorm2, residual2
                self.__sp_partition_if_applicable(sbh),
                # LAYERNORM 2
                self.__sp_partition_if_applicable(sbh),  # up/gate input
                # MLP UP/GATE (col parallel linear)
                self.__tp_partition(2 * sbi if self.glu else sbi),  # SwiGLU input
                # SwiGLU
                self.__tp_partition(sbi),  # SiLU output
                self.__tp_partition(sbi),  # gate output
                # MLP DOWN (row parallel linear)
                # TODO. why isn't this deallocated
                self.__sp_partition_if_applicable(sbh),
                # DROPOUT
                self.__sp_partition_if_applicable(0.5 * sbh),  # dropout mask
                # - output deallocated - residual doesn't need to store
                # RESIDUAL
                # input to next norm1, resid1
                self.__sp_partition_if_applicable(sbh),
            )
        elif self.act_ckpting_type == ActivationCheckpointingType.NONE:
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

    def __sp_partition_if_applicable(self, unpartitioned_numel: int) -> int:
        if not self.parallelism_cfg.sp_enabled:
            return unpartitioned_numel
        return safe_divide(unpartitioned_numel, self.parallelism_cfg.tp)
