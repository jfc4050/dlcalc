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


@dataclasses.dataclass
class ThreeDParallelModel:
    """
    a great blog post summarizing many of the equations coded below:
    https://eleutherai.notion.site/Transformers-Math-101-d2fcfc7a25d446388fde97821ad2412a

    also see:
    https://gist.github.com/Quentin-Anthony/f43939791a7ceb0b01a4937308317be5
    """

    parallelism_cfg: ParallelismConfig
    n_params: int

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

    activation_checkpointing_type: ActivationCheckpointingType

    # TODO. assuming mixed precision here.
    bytes_per_parameter: int = 2

    def get_transformer_block_n_params(self) -> int:
        numel = sum(
            [
                # qkv_proj (col parallel)
                self.hidden_sz
                * ((self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim),
                # attn out_proj (row parallel)
                self.hidden_sz * self.hidden_sz,
                # MLP layer 1 (col parallel)
                self.hidden_sz * ((self.inter_sz * 2) if self.glu else self.inter_sz),
                # MLP layer 2 (row parallel)
                self.inter_sz * self.hidden_sz,
            ]
        ) / self.parallelism_cfg.tp

        return Size(numel=numel, bytes_per_element=self.bytes_per_parameter)

    def get_embedding_or_lm_head_n_params(self) -> int:
        return Size(
            numel=self.hidden_sz * self.vocab_sz / self.parallelism_cfg.tp,
            bytes_per_element=self.bytes_per_parameter,
        )

    def get_states(self, training: bool) -> States:
        return States(
            params=Size(
                numel=self.__param_numel_per_mp_rank(),
                bytes_per_element=self.bytes_per_parameter,
            ),
            grads=Size(
                numel=self.__grad_numel_per_mp_rank(),
                bytes_per_element=self.bytes_per_parameter,
            )
            if training
            else None,
            optim_states=Size(
                numel=self.__optim_states_numel_per_mp_rank(),
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
        sp = self.parallelism_cfg.sp_enabled

        sbh = self.sequence_len * self.microbatch_sz * self.hidden_sz
        sbi = self.sequence_len * self.microbatch_sz * self.inter_sz

        if self.activation_checkpointing_type == ActivationCheckpointingType.FULL:
            # just the block input
            return safe_divide(sbh, self.parallelism_cfg.tp) if sp else sbh
        elif (
            self.activation_checkpointing_type == ActivationCheckpointingType.SELECTIVE
        ):
            sp = self.parallelism_cfg.sp_enabled
            return sum(
                [
                    # LAYERNORM 1
                    safe_divide(sbh, self.parallelism_cfg.tp) if sp else sbh,  # input
                    # ATTENTION
                    sbh,  # QKV matmul input
                    # skipping Q @ K.T (checkpointed by FlashAttention)
                    # skipping Softmax (checkpointed by FlashAttention)
                    # skipping Softmax dropout (checkpointed by FlashAttention)
                    safe_divide(sbh, self.parallelism_cfg.tp),  # down proj input.
                    # LAYERNORM 2
                    sbh,  # layernorm 2 input
                    # MLP
                    sbh,  # up/gate input
                    safe_divide(sbi, self.parallelism_cfg.tp),  # activation input
                    safe_divide(sbi, self.parallelism_cfg.tp),  # down input
                    safe_divide(sbh, 2),  # dropout mask
                    # TODO. this is sort of hacky, dropout mask is 1 byte so setting as half numel
                ]
            )
        elif self.activation_checkpointing_type == ActivationCheckpointingType.NONE:
            return sum(
                [
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
                ]
            )
        else:
            raise ValueError(
                f"unhandled checkpointing_type={self.activation_checkpointing_type}"
            )

    def __param_numel_per_mp_rank(self) -> int:
        """parameters are partitioned with tensor and pipeline parallelism. note this is
        an approximation. not all parameters are evenly partitioned."""
        if (
            self.parallelism_cfg.zero_level
            >= ParallelismConfig.ZeroLevel.PARTITION_PARAMETERS
        ):
            raise NotImplementedError

        return self.n_params / self.parallelism_cfg.mp_degree()

    def __grad_numel_per_mp_rank(self) -> int:
        """gradients are not partitioned by TP.

        see: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md
        """
        if (
            self.parallelism_cfg.zero_level
            < ParallelismConfig.ZeroLevel.PARTITION_GRADIENTS
        ):
            return self.n_params / self.parallelism_cfg.mp_degree()
        else:
            return self.n_params / self.parallelism_cfg.world_size()

    def __optim_states_numel_per_mp_rank(self) -> int:
        """here we assume Adam, that is three optimizer states per parameter -
        (param, momentum, variance)

        see https://arxiv.org/pdf/1910.02054.pdf. This equation is for ZeRO1 optimizer
        used along with 3D parallelism.

        also see https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#distributed-optimizer
        """
        numel_unpartitioned_adam = self.n_params * 3
        if (
            self.parallelism_cfg.zero_level
            < ParallelismConfig.ZeroLevel.PARTITION_OPTIMIZER
        ):
            return numel_unpartitioned_adam / self.parallelism_cfg.mp_degree()
        else:
            return (
                # fp32 optimizer state
                self.n_params / self.parallelism_cfg.mp_degree()
                # momentum + variance
                + 2 * self.n_params / self.parallelism_cfg.world_size()
            )
