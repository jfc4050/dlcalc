# mypy: disable-error-code="no-any-unimported"
import pprint
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Tuple

import torch  # type: ignore[import-untyped]
from torch import Tensor  # type: ignore[import-untyped]


@dataclass  # type: ignore[no-any-unimported]
class TensorSummary:
    shape: Tuple[int]  # type: ignore[no-any-unimported]
    dtype: torch.dtype  # type: ignore[no-any-unimported]

    @staticmethod
    def from_tensor(tensor: Tensor) -> "TensorSummary":  # type: ignore[no-any-unimported]
        return TensorSummary(tuple(tensor.shape), dtype=tensor.dtype)  # type: ignore[no-any-unimported]


def summarize(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: summarize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [summarize(o) for o in obj]
    elif isinstance(obj, Tensor):
        return TensorSummary.from_tensor(obj)
    elif hasattr(obj, "__dict__"):
        return summarize(obj.__dict__)
    else:
        return obj


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument("checkpoint_path")
    args = parser.parse_args()
    c = torch.load(args.checkpoint_path)

    summarized = summarize(c)

    pprint.pprint(summarized, width=120)


if __name__ == "__main__":
    main()
