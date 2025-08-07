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
from .model_3d import ParallelConfig


def get_tp_reduce_scatter_comm_time_s(
    size: Size, parallel_config: ParallelConfig, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size, n_participants=parallel_config.tp, machine_spec=machine_spec
    )


def get_tp_all_gather_comm_time_s(
    size: Size, parallel_config: ParallelConfig, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size, n_participants=parallel_config.tp, machine_spec=machine_spec
    )


def get_dp_reduce_scatter_latency_term_s(
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_latency_term_s(
        n_participants=parallel_config.dp,
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )


def get_dp_reduce_scatter_bw_term_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    # full BW should be divided along all MP ranks within a single node, since
    # they are each participating in their own DP collectives.
    # We assume TP is the only form of MP we do within node.
    mp_degree_in_node = parallel_config.tp
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=parallel_config.dp,
        unidirectional_link_bw_bytes_per_sec=machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec
        // mp_degree_in_node,
    )


def get_dp_reduce_scatter_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    latency_term = get_dp_reduce_scatter_latency_term_s(
        parallel_config=parallel_config,
        machine_spec=machine_spec,
    )

    bw_term = get_dp_reduce_scatter_bw_term_s(
        size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
    )

    return latency_term + bw_term


def get_dp_all_gather_latency_term_s(
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_latency_term_s(
        n_participants=parallel_config.dp,
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )


def get_dp_all_gather_bw_term_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    # full BW should be divided along all MP ranks within a single node, since
    # they are each participating in their own DP collectives.
    # We assume TP is the only form of MP we do within node.
    mp_degree_in_node = parallel_config.tp
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=parallel_config.dp,
        unidirectional_link_bw_bytes_per_sec=machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec
        // mp_degree_in_node,
    )


def get_dp_all_gather_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    latency_term = get_dp_all_gather_latency_term_s(
        parallel_config=parallel_config,
        machine_spec=machine_spec,
    )

    bw_term = get_dp_all_gather_bw_term_s(
        size,
        parallel_config=parallel_config,
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


def get_all_to_all_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    # For all-to-all in MoE context, participants are expert parallel ranks
    n_participants = parallel_config.expert_mesh.ep if parallel_config.expert_mesh else 1
    mp_degree_in_node = parallel_config.tp

    lat_term_s = machine_spec.inter_node_connect.latency_sec

    # we'll just model it as simultaneous sends of partition to all other participants.
    bw = (
        (machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec / mp_degree_in_node)
        if n_participants > 8
        else machine_spec.intra_node_connect.unidirectional_bw_bytes_per_sec
    )

    bw_term_s = ((size.bytes() // n_participants) * (n_participants - 1)) / bw

    return lat_term_s + bw_term_s


def get_expert_tp_all_gather_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """All-gather for expert tensor parallelism. Uses expert_tp degree."""
    if not parallel_config.expert_mesh:
        return 0.0
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size, n_participants=parallel_config.expert_mesh.tp, machine_spec=machine_spec
    )


def get_expert_tp_reduce_scatter_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """Reduce-scatter for expert tensor parallelism. Uses expert_tp degree."""
    if not parallel_config.expert_mesh:
        return 0.0
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size, n_participants=parallel_config.expert_mesh.tp, machine_spec=machine_spec
    )
