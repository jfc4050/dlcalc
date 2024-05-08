import pprint
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class TensorSummary:
    shape: Tuple[int]
    dtype: torch.dtype

    @staticmethod
    def from_tensor(tensor: Tensor) -> "TensorSummary":
        return TensorSummary(tuple(tensor.shape), dtype=tensor.dtype)


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


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("checkpoint_path")
    args = parser.parse_args()
    c = torch.load(args.checkpoint_path)

    summarized = summarize(c)

    pprint.pprint(summarized, width=120)


if __name__ == "__main__":
    main()
