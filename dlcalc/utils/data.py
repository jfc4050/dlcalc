import math
from typing import Tuple

from .math import product, safe_divide


class Size:
    def __init__(self, numel: int, bits_per_element: int) -> None:
        self._numel = numel
        self._bits_per_element = bits_per_element

    def numel(self) -> int:
        return self._numel

    def bits(self) -> int:
        return self._numel * self._bits_per_element

    def bytes(self) -> int:
        return self._numel * safe_divide(self._bits_per_element, 8)

    def __add__(self, other: "Size") -> "Size":
        if self._bits_per_element != other._bits_per_element:
            raise ValueError(
                f"attempted addition between two sizes with different bits per element "
                f"{self._bits_per_element} vs {other._bits_per_element}"
            )

        return Size(numel=self._numel + other._numel, bits_per_element=self._bits_per_element)

    def __mul__(self, multiplicand) -> "Size":
        return Size(self._numel * multiplicand, self._bits_per_element)

    def __rmul__(self, multiplicand) -> "Size":
        return Size(self._numel * multiplicand, self._bits_per_element)

    def __repr__(self) -> str:
        return (
            f"numel: {self._numel * 1e-9:.3f} B, "
            f"bytes: {self._numel * safe_divide(self._bits_per_element, 8) / (1024 ** 3):.3f} GiB"
        )


class TensorRepr:
    def __init__(
        self,
        unpartitioned_shape: Tuple[int, ...],
        partition_degree: int,
        bits_per_elt: int,
        enforce_evenly_partitionable: bool = True,
    ) -> None:
        self._shape = unpartitioned_shape
        self._numel = product(unpartitioned_shape)
        self._partition_degree = partition_degree
        self._bits_per_elt = bits_per_elt
        self._bytes_per_elt = safe_divide(bits_per_elt, 8)

        if enforce_evenly_partitionable and self._numel % self._partition_degree != 0:
            raise RuntimeError(f"{self._numel} not divisible by {self._partition_degree}")

    def shape(self, partitioned: bool) -> Tuple[int, ...]:
        if partitioned:
            raise NotImplementedError
        else:
            return self._shape

    def size(self, partitioned: bool) -> Size:
        return Size(
            numel=self.__get_numel(partitioned=partitioned), bits_per_element=self._bits_per_elt
        )

    def __get_numel(self, partitioned: bool) -> int:
        if partitioned:
            return int(math.ceil(self._numel / self._partition_degree))
        else:
            return self._numel
