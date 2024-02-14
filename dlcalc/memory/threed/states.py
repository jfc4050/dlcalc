import dataclasses
from enum import Enum

from dlcalc.utils.math import safe_divide
from ...utils.size import Size
from ...utils.configurations import ActivationCheckpointingType


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

    sp_enabled: bool

    zero_level: ZeroLevel

    def mp_degree(self) -> int:
        return self.pp * self.pp

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


@dataclasses.dataclass
class ThreeDParallelModel:
    """
    a great blog post summarizing many of the equations coded below:
    https://eleutherai.notion.site/Transformers-Math-101-d2fcfc7a25d446388fde97821ad2412a

    also see:
    https://gist.github.com/Quentin-Anthony/f43939791a7ceb0b01a4937308317be5
    """

    n_params: int
    parallelism_cfg: ParallelismConfig

    sequence_len: int
    microbatch_sz: int
    hidden_sz: int

    n_layers: int
    n_kv_heads: int
    head_dim: int

    activation_checkpointing_type: ActivationCheckpointingType

    # TODO. assuming mixed precision here.
    bytes_per_parameter: int = 2

    def activation_size_per_microbatch_per_layer(self) -> Size:
        # TODO. assuming half precision here.
        return Size(
            numel=self.__activation_numel_per_microbatch_per_layer(),
            bytes_per_element=self.bytes_per_parameter,
        )

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

        inter_sz = self.hidden_sz * 4  # TODO. hardcoding this for now

        sbh = self.sequence_len * self.microbatch_sz * self.hidden_sz
        sbi = self.sequence_len * self.microbatch_sz * inter_sz

        if self.activation_checkpointing_type == ActivationCheckpointingType.FULL:
            return sbh  # just the block input
        elif (
            self.activation_checkpointing_type == ActivationCheckpointingType.SELECTIVE
        ):
            if self.parallelism_cfg.sp_enabled:
                return sbh * 17 / self.parallelism_cfg.tp
            else:
                return sbh * (5 + 12 / self.parallelism_cfg.tp)
        elif self.activation_checkpointing_type == ActivationCheckpointingType.NONE:
            return sum(
                [
                    # LAYERNORM 1
                    sbh,  # layernorm 2 input
                    # ATTENTION
                    sbh,  # QKV matmul input
                    # skipping Q @ K.T (checkpointed by FlashAttention)
                    # skipping Softmax (checkpointed by FlashAttention)
                    # skipping Softmax dropout (checkpointed by FlashAttention)
                    sbh,  # attention over values.
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


class States:
    def __init__(self, params: Size, grads: Size, optim_states: Size) -> None:
        self.params = params
        self.grads = grads
        self.optim_states = optim_states

    @staticmethod
    def for_frozen_zero3_half_precision(model: Zero3Model) -> "States":
        return States(
            params=Size(
                numel=model.params_per_rank(),
                bytes_per_element=model.bytes_per_parameter,
            ),
            grads=None,
            optim_states=None,
        )

    @staticmethod
    def for_unfrozen_zero3_half_precision(model: Zero3Model) -> "States":
        return States(
            params=Size(
                numel=model.params_per_rank(),
                bytes_per_element=model.bytes_per_parameter,
            ),
            grads=Size(
                numel=model.params_per_rank(),
                bytes_per_element=model.bytes_per_parameter,
            ),
            optim_states=Size(
                numel=3 * model.params_per_rank(),
                # TODO. assuming AMP here.
                bytes_per_element=4,
            ),
        )

    @staticmethod
    def for_frozen_3d_half_precision(model: ThreeDParallelModel) -> "States":
        return States(
            params=Size(
                numel=__class__.__param_numel_per_mp_rank(model),
                bytes_per_element=model.bytes_per_parameter,
            ),
            grads=None,
            optim_states=None,
        )

    @staticmethod
    def for_unfrozen_3d_mixed_precision(model: ThreeDParallelModel) -> "States":
        return States(
            params=Size(
                numel=__class__.__param_numel_per_mp_rank(model),
                bytes_per_element=model.bytes_per_parameter,
            ),
            grads=Size(
                numel=__class__.__grad_numel_per_mp_rank(model),
                bytes_per_element=model.bytes_per_parameter,
            ),
            optim_states=Size(
                numel=__class__.__optim_states_numel_per_mp_rank(model),
                # TODO. assuming AMP here.
                bytes_per_element=4,
            ),
        )

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
                f"TOTAL        : {self.total_bytes() / (1024 ** 3):.2f}GB"
            ]
        )

    @staticmethod
    def __param_numel_per_mp_rank(model: ThreeDParallelModel) -> int:
        """parameters are partitioned with tensor and pipeline parallelism. note this is
        an approximation. not all parameters are evenly partitioned."""
        if (
            model.parallelism_cfg.zero_level
            >= ParallelismConfig.ZeroLevel.PARTITION_PARAMETERS
        ):
            raise NotImplementedError

        return model.n_params / model.parallelism_cfg.mp_degree()

    @staticmethod
    def __grad_numel_per_mp_rank(model: ThreeDParallelModel) -> int:
        """gradients are not partitioned by TP.

        see: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md
        """
        if (
            model.parallelism_cfg.zero_level
            < ParallelismConfig.ZeroLevel.PARTITION_GRADIENTS
        ):
            return model.n_params / model.parallelism_cfg.mp_degree()
        else:
            if model.parallelism_cfg.pp > 1:
                raise RuntimeError(
                    "While it's theoretically possible to use ZeRO stage 2 with "
                    "Pipeline Parallelism, it will have bad performance impacts. "
                    "There would need to be an additional reduce-scatter collective "
                    "for every micro-batch to aggregate the gradients "
                    "before sharding, which adds a potentially significant communication overhead. "
                    "By nature of Pipeline Parallelism, small micro-batches are used and instead "
                    "the focus is on trying to balance arithmetic intensity (micro-batch size) "
                    "with minimizing the Pipeline bubble (number of micro-batches). "
                    "Therefore those communication costs are going to hurt."
                )

            return model.n_params / model.parallelism_cfg.world_size()

    @staticmethod
    def __optim_states_numel_per_mp_rank(
        model: ThreeDParallelModel,
    ) -> int:
        """here we assume Adam, that is three optimizer states per parameter -
        (param, momentum, variance)

        see https://arxiv.org/pdf/1910.02054.pdf. This equation is for ZeRO1 optimizer
        used along with 3D parallelism.

        also see https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#distributed-optimizer
        """
        numel_unpartitioned_adam = model.n_params * 3
        if (
            model.parallelism_cfg.zero_level
            < ParallelismConfig.ZeroLevel.PARTITION_OPTIMIZER
        ):
            return numel_unpartitioned_adam / model.parallelism_cfg.mp_degree()
        else:
            return numel_unpartitioned_adam / model.parallelism_cfg.world_size()
