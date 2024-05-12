from typing import Tuple


def safe_divide(numerator: int, denominator: int) -> int:
    """return numerator / denominator, or throw an exception if its not divisible."""
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} not divisible by {denominator}")

    return numerator // denominator


def product(multiplicands: Tuple[int, ...]) -> int:
    """return the product of a sequence of multiplicands."""
    assert len(multiplicands) > 0
    product = 1
    for multiplicand in multiplicands:
        product *= multiplicand

    return product


def compute_gemm_flops(weight_shape: Tuple[int, ...], seqlen: int, batch_sz: int) -> float:
    """compute the number of FLOPs in a linear layer. Given by 2MNK."""
    return 2 * batch_sz * seqlen * product(weight_shape)
