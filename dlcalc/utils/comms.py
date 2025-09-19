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
from .hardware import LinkSpec, MachineSpec
from .model_3d import ParallelConfig


class ParallelismType(Enum):
    """Types of parallelism in hierarchical order."""

    PP = 0  # Pipeline Parallel
    DP = 1  # Data Parallel
    CP = 2  # Context Parallel
    TP = 3  # Tensor Parallel

    # Expert parallelism types
    EDP = 4  # Expert Data Parallel
    EP = 5  # Expert Parallel
    ETP = 6  # Expert Tensor Parallel


def _get_effective_link_spec(
    *,
    parallelism_type: ParallelismType,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool,
) -> LinkSpec:
    """Calculate effective link specification for a given parallelism type.

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

    comm_has_internode_component = product_including_current > n_devices_per_node

    return (
        machine_spec.inter_node_connect
        if comm_has_internode_component
        else machine_spec.intra_node_connect
    )


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


def get_cp_ring_exchange_comm_time_s(
    size: Size, parallel_config: ParallelConfig, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""

    effective_link_spec = _get_effective_link_spec(
        parallelism_type=ParallelismType.CP,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=False,
    )

    latency_term_s = _ring_ag_or_rs_latency_term_s(
        n_participants=parallel_config.cp,
        link_latency_s=effective_link_spec.latency_sec,
    )

    bw_term_s = _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=parallel_config.cp,
        unidirectional_link_bw_bytes_per_sec=effective_link_spec.unidirectional_bw_bytes_per_sec,
    )

    return latency_term_s + bw_term_s


def get_pp_sendrecv_comm_time_s(
    size: Size, parallel_config: ParallelConfig, machine_spec: MachineSpec
) -> float:
    effective_link_spec = _get_effective_link_spec(
        parallelism_type=ParallelismType.PP,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=False,
    )

    latency_term_s = effective_link_spec.latency_sec
    bw_term_s = size.bytes() / effective_link_spec.unidirectional_bw_bytes_per_sec

    return latency_term_s + bw_term_s


def get_dp_reduce_scatter_latency_term_s(
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    effective_link_spec = _get_effective_link_spec(
        parallelism_type=ParallelismType.DP
        if parallel_config.expert_mesh is None
        else ParallelismType.EDP,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=parallel_config.expert_mesh is not None,
    )
    return _ring_ag_or_rs_latency_term_s(
        # approximation: we assume most of the parameters are MoE parameters, and calculate
        # just for those. for most models this is good enough.
        n_participants=parallel_config.dp
        if parallel_config.expert_mesh is None
        else parallel_config.expert_mesh.dp,
        link_latency_s=effective_link_spec.latency_sec,
    )


def get_dp_reduce_scatter_bw_term_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    # approximation: we assume most of the parameters are MoE parameters, and calculate
    # just for those. for most models this is good enough.
    effective_link_spec = _get_effective_link_spec(
        parallelism_type=ParallelismType.DP
        if parallel_config.expert_mesh is None
        else ParallelismType.EDP,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=parallel_config.expert_mesh is not None,
    )
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=parallel_config.dp
        if parallel_config.expert_mesh is None
        else parallel_config.expert_mesh.dp,
        unidirectional_link_bw_bytes_per_sec=effective_link_spec.unidirectional_bw_bytes_per_sec,
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
    effective_link_spec = _get_effective_link_spec(
        parallelism_type=ParallelismType.DP
        if parallel_config.expert_mesh is None
        else ParallelismType.EDP,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=parallel_config.expert_mesh is not None,
    )
    return _ring_ag_or_rs_latency_term_s(
        # approximation: we assume most of the parameters are MoE parameters, and calculate
        # just for those. for most models this is good enough.
        n_participants=parallel_config.dp
        if parallel_config.expert_mesh is None
        else parallel_config.expert_mesh.dp,
        link_latency_s=effective_link_spec.latency_sec,
    )


def get_dp_all_gather_bw_term_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """assumes ring algorithm."""
    effective_link_spec = _get_effective_link_spec(
        parallelism_type=ParallelismType.DP
        if parallel_config.expert_mesh is None
        else ParallelismType.EDP,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=parallel_config.expert_mesh is not None,
    )
    # approximation: we assume most of the parameters are MoE parameters, and calculate
    # just for those. for most models this is good enough.
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=parallel_config.dp
        if parallel_config.expert_mesh is None
        else parallel_config.expert_mesh.dp,
        unidirectional_link_bw_bytes_per_sec=effective_link_spec.unidirectional_bw_bytes_per_sec,
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
    effective_link_spec = _get_effective_link_spec(
        parallelism_type=ParallelismType.ETP if is_expert_comm else ParallelismType.TP,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=is_expert_comm,
    )
    lat_term_s = _ring_ag_or_rs_latency_term_s(
        n_participants=n_participants,
        link_latency_s=effective_link_spec.latency_sec,
    )
    bw_term_s = _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=n_participants,
        unidirectional_link_bw_bytes_per_sec=effective_link_spec.unidirectional_bw_bytes_per_sec,
    )

    return bw_term_s + lat_term_s


def get_all_to_all_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    assert parallel_config.expert_mesh is not None

    alltoall_n_nodes = parallel_config.expert_mesh.ep // (
        machine_spec.n_devices // parallel_config.expert_mesh.tp
    )

    lat_term_s = (
        machine_spec.inter_node_connect.latency_sec
        if alltoall_n_nodes > 1
        else machine_spec.intra_node_connect.latency_sec
    )

    # we need to calculate the percentage of the comms that occur over the slowest link
    # type, and only calculate the comm cost for the percentage of the message that
    # travels over the slowest links.
    #
    # this is kind of a special situation for all-to-all. the other comm types predominantly
    # use rings, where the slowest links act as a dam that rate limit the rest.
    n_participants = parallel_config.expert_mesh.ep
    if alltoall_n_nodes > 1:
        alltoall_internode_fraction = (alltoall_n_nodes - 1) / alltoall_n_nodes
        bw_term_s = (
            (size.bytes() / n_participants) * alltoall_internode_fraction * (n_participants - 1)
        ) / machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec
    else:
        bw_term_s = (
            (size.bytes() / n_participants) * (n_participants - 1)
        ) / machine_spec.intra_node_connect.unidirectional_bw_bytes_per_sec

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
        machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec,
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
