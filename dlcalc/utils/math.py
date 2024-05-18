import math
from typing import Tuple


def safe_divide(numerator: int, denominator: int) -> int:
    """return numerator / denominator, or throw an exception if its not divisible."""
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} not divisible by {denominator}")

    return numerator // denominator


def ceil_divide(numerator: int, denominator: int) -> int:
    """return ceil(numerator / denominator)"""
    return int(math.ceil(numerator / denominator))


def product(multiplicands: Tuple[int, ...]) -> int:
    """return the product of a sequence of multiplicands."""
    assert len(multiplicands) > 0
    product = 1
    for multiplicand in multiplicands:
        product *= multiplicand

    return product
