from typing import Tuple


def safe_divide(numerator: int, denominator: int) -> int:
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} not divisible by {denominator}")

    return numerator // denominator


def product(multiplicands: Tuple[int, ...]) -> int:
    assert len(multiplicands) > 0
    product = 1
    for multiplicand in multiplicands:
        product *= multiplicand

    return product


def compute_gemm_flops(weight_shape: Tuple[int, ...], seqlen: int, batch_sz: int) -> float:
    return 2 * batch_sz * seqlen * product(weight_shape)
