import abc
from abc import ABC
from datetime import timedelta
from enum import Enum

from dlcalc.utils.hardware import MachineSpec
from dlcalc.utils.math import safe_divide


class Dtype(Enum):
    pass


class Op(ABC):
    @abc.abstractmethod
    def compute_runtime_s(self, machine_spec: MachineSpec) -> timedelta:
        raise NotImplementedError

    @abc.abstractmethod
    def get_n_params(self, partitioned: bool) -> int:
        raise NotImplementedError


class Transformer(Op):
    pass


class TransformerBlock(Op):
    pass


class Attention(Op):
    pass


class MLP(Op):
    pass


class ExpertMLP(Op):
    def __init__(
        self,
        n_tokens: int,
        n_experts: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
        glu: bool,
        ep: int,
        tp: int,
    ) -> None:
        expert_capacity = None  # TODO.

        self.router = GEMM(
            m=n_tokens,
            k=hidden_dim,
            n=n_experts,
        )

        self.mlp_up = BatchedGEMM(
            m=expert_capacity,
            k=hidden_dim,
            # following common practice of merging up + gate matmuls in the event
            # we're using GLU.
            n=(mlp_hidden_dim * 2) if glu else mlp_hidden_dim,
            m_partition_degree=None,  # TODO.
            k_partition_degree=1,
            n_partition_degree=tp,
        )
        self.mlp_down = BatchedGEMM(
            m=expert_capacity,
            k=(mlp_hidden_dim * 2) if glu else mlp_hidden_dim,
            n=hidden_dim,
        )


class Norm(Op):
    def __init__(self, n_tokens: int, hidden_dim: int, n_tokens_partition_degree: int) -> None:
        self.__n_tokens = n_tokens
        self.__hidden_dim = hidden_dim
        self.__n_tokens_partition_degree = n_tokens_partition_degree

    def compute_runtime_s(self, machine_spec: MachineSpec) -> timedelta:
        pass


class SDPA(Op):
    pass


class GEMM(Op):
    def __init__(
        self,
        m: int,
        k: int,
        n: int,
        m_partition_degree: int,
        k_partition_degree: int,
        n_partition_degree: int,
        dtype: Dtype,
    ) -> None:
        self.__m = m
        self.__k = k
        self.__n = n
        self.__m_partition_degree = m_partition_degree
        self.__k_partition_degree = k_partition_degree
        self.__n_partition_degree = n_partition_degree

    def compute_runtime_s(self, machine_spec: MachineSpec) -> timedelta:
        m_partitioned = safe_divide(self.__m, self.__m_partition_degree)
        n_partitioned = safe_divide(self.__n, self.__n_partition_degree)
        k_partitioned = safe_divide(self.__k, self.__k_partition_degree)

        n_flops_partitioned = m_partitioned * n_partitioned * k_partitioned

        n_secs = n_flops_partitioned / machine_spec.device_spec.peak_flops

        return timedelta.seconds(n_secs)

    def get_n_params(self, partitioned: bool) -> int:
        if partitioned:
            k_partitioned = safe_divide(self.__k, self.__k_partition_degree)
            n_partitioned = safe_divide(self.__n, self.__n_partition_degree)
            return k_partitioned * n_partitioned
        else:
            return self.__k * self.__n


class BatchedGEMM(Op):
    def __init__(self, b: int, m: int, k: int, n: int) -> None:
        self.b = b
        self.m = m
        self.k = k
        self.n = n
