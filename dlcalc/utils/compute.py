"""
https://arxiv.org/pdf/2401.14489
https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
"""

from dlcalc.utils.data import TensorRepr
from dlcalc.utils.hardware import MachineSpec

from .math import product


def compute_gemm_flops(n_tokens: int, weight_shape: tuple[int, ...]) -> float:
    """compute the number of FLOPs in a linear layer. Given by 2MNK."""
    return 2 * n_tokens * product(*weight_shape)


def gemm_time_s(
    *,
    n_tokens: int,
    weight_repr: TensorRepr,
    machine_spec: MachineSpec,
) -> float:
    weight_shape = weight_repr.shape(partitioned=True)
    flops = compute_gemm_flops(n_tokens=n_tokens, weight_shape=weight_shape)
    compute_time = flops / (machine_spec.device_spec.peak_flops * machine_spec.device_spec.mamf)

    bytes_per_elt = weight_repr._bits_per_elt // 8
    gemm_k, gemm_n = weight_shape
    total_bytes = (n_tokens * gemm_k + gemm_k * gemm_n + n_tokens * gemm_n) * bytes_per_elt
    memory_time = total_bytes / machine_spec.device_spec.mem_bandwidth_bytes_per_sec

    return max(compute_time, memory_time)


def expert_gemm_time_s(
    *,
    n_tokens_per_expert: int,
    weight_repr: TensorRepr,
    machine_spec: MachineSpec,
) -> float:
    n_local_experts, *gemm_dims = weight_repr.shape(partitioned=True)
    flops = n_local_experts * compute_gemm_flops(
        n_tokens=n_tokens_per_expert,
        weight_shape=tuple(gemm_dims),
    )
    compute_time = flops / (machine_spec.device_spec.peak_flops * machine_spec.device_spec.mamf)

    bytes_per_elt = weight_repr._bits_per_elt // 8
    gemm_k, gemm_n = gemm_dims
    per_expert_bytes = (
        n_tokens_per_expert * gemm_k + gemm_k * gemm_n + n_tokens_per_expert * gemm_n
    ) * bytes_per_elt
    total_bytes = n_local_experts * per_expert_bytes
    memory_time = total_bytes / machine_spec.device_spec.mem_bandwidth_bytes_per_sec

    return max(compute_time, memory_time)


def sdpa_time_s(
    *,
    n_tokens: int,
    ctxlen: int,
    n_heads: int,
    head_dim: int,
    machine_spec: MachineSpec,
) -> float:
    return (
        n_tokens
        * n_heads
        * sum(
            [
                2 * head_dim * ctxlen,  # Q @ K.T
                2 * ctxlen * head_dim,  # A @ V
            ]
        )
        / (machine_spec.device_spec.peak_flops * machine_spec.device_spec.mamf)
    )
