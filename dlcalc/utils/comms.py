"""
we assume rings here but this may not always be the case.
see: https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/
"""

from .data import Size
from .hardware import MachineSpec


def _ring_ag_or_rs_comm_time_s(
    size: Size,
    n_participants: int,
    # NOTE: assumes duplex bw = 2x unidirectional
    unidirectional_link_bw_bytes_per_sec: int,
    link_latency_s: float,
) -> float:
    phase_sent_data_size_bytes = int(size.bytes() / n_participants)

    n_phases = n_participants - 1

    phase_bw_term_s = phase_sent_data_size_bytes / unidirectional_link_bw_bytes_per_sec
    phase_lat_term_s = link_latency_s

    return n_phases * (phase_bw_term_s + phase_lat_term_s)


def get_tp_reduce_scatter_comm_time_s(
    size: Size, n_participants: int, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_comm_time_s(
        size,
        n_participants=n_participants,
        unidirectional_link_bw_bytes_per_sec=machine_spec.intra_node_connect.unidirectional_bw_bytes_per_sec,
        link_latency_s=machine_spec.intra_node_connect.latency_sec,
    )


def get_tp_all_gather_comm_time_s(
    size: Size, n_participants: int, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_comm_time_s(
        size,
        n_participants=n_participants,
        unidirectional_link_bw_bytes_per_sec=machine_spec.intra_node_connect.unidirectional_bw_bytes_per_sec,
        link_latency_s=machine_spec.intra_node_connect.latency_sec,
    )


def get_dp_reduce_scatter_comm_time_s(
    size: Size,
    n_participants: int,
    mp_degree_in_node: int,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_comm_time_s(
        size,
        n_participants=n_participants,
        # full BW should be divided along all MP ranks within a single node, since
        # they are each participating in their own DP collectives.
        unidirectional_link_bw_bytes_per_sec=int(
            machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec / mp_degree_in_node
        ),
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )


def get_dp_all_gather_comm_time_s(
    size: Size,
    n_participants: int,
    mp_degree_in_node: int,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_comm_time_s(
        size,
        n_participants=n_participants,
        # full BW should be divided along all MP ranks within a single node, since
        # they are each participating in their own DP collectives.
        unidirectional_link_bw_bytes_per_sec=int(
            machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec / mp_degree_in_node
        ),
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )
