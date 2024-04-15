"""
we assume rings here but this may not always be the case.
see: https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/
"""

from .data import Size


def get_reduce_scatter_comm_volume(size: Size, n_participants: int) -> Size:
    """assumes ring algorithm."""
    n = size
    p = n_participants
    return n * ((p - 1) / p)


def get_all_gather_comm_volume(size: Size, n_participants: int) -> Size:
    """assumes ring algorithm."""
    n = size
    p = n_participants
    return n * ((p - 1) / p)
