"""
Reduce-Scatter:
* Ring: https://github.com/NVIDIA/nccl/blob/ab2b89c4c339bd7f816fbc114a4b05d386b66290/src/device/reduce_scatter.h#L12-L54

All-Gather:
* Ring: https://github.com/NVIDIA/nccl/blob/ab2b89c4c339bd7f816fbc114a4b05d386b66290/src/device/all_gather.h#L12-L64

All-Reduce (not relevant for comm patterns we have represented for now):
some info on tree AR: https://github.com/NVIDIA/nccl/issues/545

TODO. try to get estimates closer by accounting for NCCL protocols https://github.com/NVIDIA/nccl/issues/281
"""

from .data import Size
from .hardware import MachineSpec


def get_tp_reduce_scatter_comm_time_s(
    size: Size, n_participants: int, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size, n_participants=n_participants, machine_spec=machine_spec
    )


def get_tp_all_gather_comm_time_s(
    size: Size, n_participants: int, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size, n_participants=n_participants, machine_spec=machine_spec
    )


def get_dp_reduce_scatter_latency_term_s(
    n_participants: int,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_latency_term_s(
        n_participants=n_participants,
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )


def get_dp_reduce_scatter_bw_term_s(
    size: Size,
    n_participants: int,
    mp_degree_in_node: int,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=n_participants,
        # full BW should be divided along all MP ranks within a single node, since
        # they are each participating in their own DP collectives.
        unidirectional_link_bw_bytes_per_sec=machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec
        // mp_degree_in_node,
    )


def get_dp_reduce_scatter_comm_time_s(
    size: Size,
    n_participants: int,
    mp_degree_in_node: int,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    latency_term = get_dp_reduce_scatter_latency_term_s(
        n_participants=n_participants,
        machine_spec=machine_spec,
    )

    bw_term = get_dp_reduce_scatter_bw_term_s(
        size,
        n_participants=n_participants,
        mp_degree_in_node=mp_degree_in_node,
        machine_spec=machine_spec,
    )

    return latency_term + bw_term


def get_dp_all_gather_latency_term_s(
    n_participants: int,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_latency_term_s(
        n_participants=n_participants,
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )


def get_dp_all_gather_bw_term_s(
    size: Size,
    n_participants: int,
    mp_degree_in_node: int,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=n_participants,
        # full BW should be divided along all MP ranks within a single node, since
        # they are each participating in their own DP collectives.
        unidirectional_link_bw_bytes_per_sec=machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec
        // mp_degree_in_node,
    )


def get_dp_all_gather_comm_time_s(
    size: Size,
    n_participants: int,
    mp_degree_in_node: int,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    latency_term = get_dp_reduce_scatter_latency_term_s(
        n_participants=n_participants,
        machine_spec=machine_spec,
    )

    bw_term = get_dp_reduce_scatter_bw_term_s(
        size,
        n_participants=n_participants,
        mp_degree_in_node=mp_degree_in_node,
        machine_spec=machine_spec,
    )

    return latency_term + bw_term


def _ring_ag_or_rs_latency_term_s(
    n_participants: int,
    link_latency_s: float,
) -> float:
    return (n_participants - 1) * link_latency_s


def _ring_ag_or_rs_bw_term_s(
    size: Size,
    n_participants: int,
    # NOTE: assumes duplex bw = 2x unidirectional
    unidirectional_link_bw_bytes_per_sec: int,
) -> float:
    phase_sent_data_size_bytes = int(size.bytes() / n_participants)
    return (n_participants - 1) * (
        phase_sent_data_size_bytes / unidirectional_link_bw_bytes_per_sec
    )


def _get_ring_tp_ag_or_rs_comm_time_s(
    size: Size,
    n_participants: int,
    machine_spec: MachineSpec,
) -> float:
    lat_term_s = _ring_ag_or_rs_latency_term_s(
        n_participants=n_participants,
        link_latency_s=machine_spec.intra_node_connect.latency_sec,
    )
    bw_term_s = _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=n_participants,
        unidirectional_link_bw_bytes_per_sec=machine_spec.intra_node_connect.unidirectional_bw_bytes_per_sec,
    )

    return bw_term_s + lat_term_s
