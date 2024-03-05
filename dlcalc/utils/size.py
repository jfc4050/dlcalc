class Size:
    def __init__(self, numel: int, bytes_per_element: int) -> None:
        self._numel = numel
        self._bytes_per_element = bytes_per_element

    def __add__(self, other: "Size") -> "Size":
        if self._bytes_per_element != other._bytes_per_element:
            raise ValueError(
                f"attempted addition between two sizes with different bytes per element "
                f"{self._bytes_per_element} vs {other._bytes_per_element}"
            )

        return Size(
            numel=self._numel + other._numel, bytes_per_element=self._bytes_per_element
        )

    def __mul__(self, multiplicand) -> "Size":
        return Size(self._numel * multiplicand, self._bytes_per_element)

    def __rmul__(self, multiplicand) -> "Size":
        return Size(self._numel * multiplicand, self._bytes_per_element)

    def __repr__(self) -> str:
        return f"numel: {self._numel * 1e-9:.3f}B, bytes: {self._numel * self._bytes_per_element / (1024 ** 3):.3f}GiB"

    def numel(self) -> int:
        return self._numel

    def bytes(self) -> int:
        return self._numel * self._bytes_per_element
