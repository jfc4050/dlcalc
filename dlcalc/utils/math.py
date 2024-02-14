def safe_divide(numerator: int, denominator: int) -> int:
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} not divisible by {denominator}")

    return numerator // denominator
