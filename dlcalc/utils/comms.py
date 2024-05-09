"""
we assume rings here but this may not always be the case.
see: https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/
"""

from .data import Size
from .hardware import MachineSpec


def _ring_ag_or_rs_comm_time_s(size: Size, n_participants: int, machine_spec: MachineSpec) -> float:
    phase_sent_data_size_bits = int(size.bits() / n_participants)

    n_phases = n_participants - 1

    # divide full BW among devices (which will be part of different DP groups)
    # TODO. this is a safe assumption vast majority of the time but might
    # want to handle when this isn't the case.
    # NOTE: assumes duplex bw = 2x unidirectional
    phase_bw_term_s = phase_sent_data_size_bits / (
        machine_spec.inter_node_connect.unidirectional_bw_bits_per_sec / machine_spec.n_devices
    )
    phase_lat_term_s = machine_spec.inter_node_connect.latency_sec

    return n_phases * (phase_bw_term_s + phase_lat_term_s)


def get_reduce_scatter_comm_time_s(
    size: Size, n_participants: int, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_comm_time_s(
        size, n_participants=n_participants, machine_spec=machine_spec
    )


def get_all_gather_comm_time_s(size: Size, n_participants: int, machine_spec: MachineSpec) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_comm_time_s(
        size, n_participants=n_participants, machine_spec=machine_spec
    )
