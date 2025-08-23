"""
Reduce-Scatter:
* Ring: https://github.com/NVIDIA/nccl/blob/ab2b89c4c339bd7f816fbc114a4b05d386b66290/src/device/reduce_scatter.h#L12-L54

All-Gather:
* Ring: https://github.com/NVIDIA/nccl/blob/ab2b89c4c339bd7f816fbc114a4b05d386b66290/src/device/all_gather.h#L12-L64

All-Reduce (not relevant for comm patterns we have represented for now):
some info on tree AR: https://github.com/NVIDIA/nccl/issues/545

TODO. try to get estimates closer by accounting for NCCL protocols https://github.com/NVIDIA/nccl/issues/281
"""

from enum import Enum

from .configurations import CrossDCConfig
from .data import Size
from .hardware import MachineSpec
from .model_3d import ParallelConfig


class ParallelismType(Enum):
    """Types of parallelism in hierarchical order."""

    PP = 0  # Pipeline Parallel
    DP = 1  # Data Parallel
    CP = 2  # Context Parallel
    TP = 3  # Tensor Parallel
    # Expert parallelism types
    EDP = 1  # Expert Data Parallel (same level as DP)
    EP = 2  # Expert Parallel (same level as CP)
    ETP = 3  # Expert Tensor Parallel (same level as TP)


def _get_effective_bw(
    *,
    parallelism_type: ParallelismType,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool,
) -> float:
    """Calculate effective bandwidth for a given parallelism type.

    Few things we need to figure out:

    Which kind of link the communication will utilize.
    -------------------------------------------------------------------------------------------
    Given some non-overlapping parallelism order like [PP, DP, CP, TP] or [PP, eDP, EP, eTP]
    if the product of parallelisms including and after the parallelism in question is less than
    or equal to the number of workers per node, then we will use intra-node links. Otherwise
    we will use inter-node links.

    How many participants will share the link.
    -------------------------------------------------------------------------------------------
    If the products of the parallelisms coming after the parallelism in question is greater than
    or equal to the number of workers per node, then the cross-node bandwidth is divided by the
    number of workers per node. Otherwise, it is divided by the number of workers per node divided
    product of following parallelisms.

    NOTE: we only account for link being shared across one communication type at a time. We do
    not account for cross-parallelism competition (like for example overlapped DP comms at the
    same time as cross-node EP comms).

    Args:
        parallelism_type: The type of parallelism for which to calculate bandwidth
        parallel_config: The parallelism configuration
        machine_spec: The machine specification
        is_expert_comm: Whether this is for expert parallelism communication

    Returns:
        Effective bandwidth in bytes per second
    """
    n_devices_per_node = machine_spec.n_devices

    if is_expert_comm:
        assert parallel_config.expert_mesh is not None
        parallelism_values = {
            ParallelismType.PP: parallel_config.pp,
            ParallelismType.EDP: parallel_config.expert_mesh.dp,
            ParallelismType.EP: parallel_config.expert_mesh.ep,
            ParallelismType.ETP: parallel_config.expert_mesh.tp,
        }
        hierarchy = [
            ParallelismType.PP,
            ParallelismType.EDP,
            ParallelismType.EP,
            ParallelismType.ETP,
        ]
    else:
        parallelism_values = {
            ParallelismType.PP: parallel_config.pp,
            ParallelismType.DP: parallel_config.dp,
            ParallelismType.CP: parallel_config.cp,
            ParallelismType.TP: parallel_config.tp,
        }
        hierarchy = [ParallelismType.PP, ParallelismType.DP, ParallelismType.CP, ParallelismType.TP]

    if parallelism_type not in hierarchy:
        raise AssertionError(
            f"Parallelism type {parallelism_type} not found in hierarchy {hierarchy}."
        )

    product_including_current = parallelism_values[parallelism_type]
    product_after_current = 1
    for parallelism_type in hierarchy[hierarchy.index(parallelism_type) + 1 :]:
        product_including_current *= parallelism_values[parallelism_type]
        product_after_current *= parallelism_values[parallelism_type]

    if product_including_current <= n_devices_per_node:  # use intra-node link (assume no sharing)
        return machine_spec.intra_node_connect.unidirectional_bw_bytes_per_sec

    # otherwise, use inter-node link
    base_bw = machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec
    # TODO. what if we have something like TP2? there would be 4 TP groups per node.
    n_groups_per_node = n_devices_per_node

    return base_bw / n_groups_per_node


def get_tp_reduce_scatter_comm_time_s(
    size: Size, parallel_config: ParallelConfig, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size,
        n_participants=parallel_config.tp,
        machine_spec=machine_spec,
        parallel_config=parallel_config,
        is_expert_comm=False,
    )


def get_tp_all_gather_comm_time_s(
    size: Size, parallel_config: ParallelConfig, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size,
        n_participants=parallel_config.tp,
        machine_spec=machine_spec,
        parallel_config=parallel_config,
        is_expert_comm=False,
    )


def get_dp_reduce_scatter_latency_term_s(
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    return _ring_ag_or_rs_latency_term_s(
        # approximation: we assume most of the parameters are MoE parameters, and calculate
        # just for those. for most models this is good enough.
        n_participants=parallel_config.dp
        if parallel_config.expert_mesh is None
        else parallel_config.expert_mesh.dp,
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )


def get_dp_reduce_scatter_bw_term_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    # approximation: we assume most of the parameters are MoE parameters, and calculate
    # just for those. for most models this is good enough.
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=parallel_config.dp
        if parallel_config.expert_mesh is None
        else parallel_config.expert_mesh.dp,
        unidirectional_link_bw_bytes_per_sec=int(
            _get_effective_bw(
                parallelism_type=ParallelismType.DP
                if parallel_config.expert_mesh is None
                else ParallelismType.EDP,
                parallel_config=parallel_config,
                machine_spec=machine_spec,
                is_expert_comm=False,
            )
        ),
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
        # approximation: we assume most of the parameters are MoE parameters, and calculate
        # just for those. for most models this is good enough.
        n_participants=parallel_config.dp
        if parallel_config.expert_mesh is None
        else parallel_config.expert_mesh.dp,
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )


def get_dp_all_gather_bw_term_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    # approximation: we assume most of the parameters are MoE parameters, and calculate
    # just for those. for most models this is good enough.
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=parallel_config.dp
        if parallel_config.expert_mesh is None
        else parallel_config.expert_mesh.dp,
        unidirectional_link_bw_bytes_per_sec=int(
            _get_effective_bw(
                parallelism_type=ParallelismType.DP
                if parallel_config.expert_mesh is None
                else ParallelismType.EDP,
                parallel_config=parallel_config,
                machine_spec=machine_spec,
                is_expert_comm=False,
            )
        ),
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
    parallel_config: ParallelConfig,
    is_expert_comm: bool = False,
) -> float:
    lat_term_s = _ring_ag_or_rs_latency_term_s(
        n_participants=n_participants,
        # TODO. need to fix
        link_latency_s=machine_spec.intra_node_connect.latency_sec,
    )
    bw_term_s = _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=n_participants,
        unidirectional_link_bw_bytes_per_sec=int(
            _get_effective_bw(
                parallelism_type=ParallelismType.ETP if is_expert_comm else ParallelismType.TP,
                parallel_config=parallel_config,
                machine_spec=machine_spec,
                is_expert_comm=is_expert_comm,
            )
        ),
    )

    return bw_term_s + lat_term_s


def get_all_to_all_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    # For all-to-all in MoE context, participants are expert parallel ranks
    n_participants = parallel_config.expert_mesh.ep if parallel_config.expert_mesh else 1

    lat_term_s = machine_spec.inter_node_connect.latency_sec

    bw = _get_effective_bw(
        parallelism_type=ParallelismType.EP,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=True,
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
        size,
        n_participants=parallel_config.expert_mesh.tp,
        machine_spec=machine_spec,
        parallel_config=parallel_config,
        is_expert_comm=True,
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
        size,
        n_participants=parallel_config.expert_mesh.tp,
        machine_spec=machine_spec,
        parallel_config=parallel_config,
        is_expert_comm=True,
    )


def get_cross_dc_dp_all_gather_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    cross_dc_config: CrossDCConfig,
) -> float:
    """Calculate cross-DC data parallel all-gather time.

    Args:
        size: Size of data to communicate
        parallel_config: Parallelism configuration
        machine_spec: Machine/hardware specification
        cross_dc_config: Cross-DC configuration

    Returns:
        Time in seconds for cross-DC all-gather
    """
    return _get_cross_dc_dp_all_gather_or_reduce_scatter_comm_time_s(
        size=size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        cross_dc_config=cross_dc_config,
    )


def get_cross_dc_dp_reduce_scatter_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    cross_dc_config: CrossDCConfig,
) -> float:
    """Calculate cross-DC data parallel reduce-scatter time.

    Args:
        size: Size of data to communicate
        parallel_config: Parallelism configuration
        machine_spec: Machine/hardware specification
        cross_dc_config: Cross-DC configuration

    Returns:
        Time in seconds for cross-DC reduce-scatter
    """
    return _get_cross_dc_dp_all_gather_or_reduce_scatter_comm_time_s(
        size=size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        cross_dc_config=cross_dc_config,
    )


def _get_cross_dc_dp_all_gather_or_reduce_scatter_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    cross_dc_config: CrossDCConfig,
) -> float:
    if parallel_config.expert_mesh is None:
        dp_degree = parallel_config.dp
    else:
        # we'll just use the expert DP case to approximate everything as
        # it'll be the most taxing (most simultaneous rings)
        dp_degree = parallel_config.expert_mesh.dp
    n_dp_rings = parallel_config.world_size() // dp_degree
    cross_dc_bw_per_ring = cross_dc_config.interconnect_bandwidth_bytes_per_sec() / n_dp_rings

    # cross-DC DP means creating heterogeneous rings, where the majority of links
    # are fast(er) inter-node links, and some of them are slower cross-DC links.
    # we use the lowest BW link when calculating how long ring collectives take.
    # this is because we expect the slowest link to behave like a dam, and
    # rate limit any faster links that come after it.
    effective_bw = min(
        cross_dc_bw_per_ring,
        machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec / machine_spec.n_devices,
    )
    bw_term = _ring_ag_or_rs_bw_term_s(
        size=size,
        n_participants=dp_degree,
        unidirectional_link_bw_bytes_per_sec=int(effective_bw),
    )

    # in a ring, there are (dp_degree - 1) total hops
    # n_dcs of these are inter-DC hops, the rest are intra-DC hops
    inter_dc_latency_term = cross_dc_config.n_dcs * cross_dc_config.interconnect_latency_s
    intra_dc_latency_term = (dp_degree - 1 - cross_dc_config.n_dcs) * (
        machine_spec.inter_node_connect.latency_sec
    )
    latency_term = inter_dc_latency_term + intra_dc_latency_term

    return latency_term + bw_term
