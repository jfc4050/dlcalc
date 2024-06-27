"""
determine the best instance placement and rank assignment to ensure DP ring
communication can occur with the fewest possible network hops.
"""

import dataclasses
import functools
import itertools
from argparse import ArgumentParser
from typing import Dict, List, Set

import boto3

import dlcalc.utils.cluster.ec2
from dlcalc.utils.math import safe_divide


@dataclasses.dataclass
class TreeNode:
    node_id: str
    parent: "TreeNode"
    children: Set["TreeNode"]

    def __hash__(self) -> int:
        return hash(self.node_id)


def dfs_tree_leaves_only(root: TreeNode) -> List[TreeNode]:
    if not root.children:
        return [root]

    list_of_list_of_leaves = []
    for child in root.children:
        list_of_list_of_leaves.append(dfs_tree_leaves_only(child))

    # now we have a list of list of leaves acquired from each subtree.
    # we want this list of lists of leaves to be ordered from largest list -> smallest list
    # this means we use the largest subtrees first - meaning that we will prioritize
    # groups of instances that can communicate with maximum "locality"
    list_of_list_of_leaves.sort(key=lambda l: len(l), reverse=True)

    # stick everything together. this gives a prioritized order of instances
    # that should be part of the same ring.
    list_of_leaves = list(itertools.chain.from_iterable(list_of_list_of_leaves))

    return list_of_leaves


@functools.lru_cache(maxsize=None)
def distance_between_nodes(node_a: TreeNode, node_b: TreeNode) -> int:
    if node_a == node_b:
        return 0

    return 2 + distance_between_nodes(node_a.parent, node_b.parent)


def traversal_distance_of_ring(ring_participants: List[TreeNode]) -> int:
    traversed_distance = 0
    for src_node_idx in range(len(ring_participants)):
        dst_node_idx = (src_node_idx + 1) % len(ring_participants)

        src_node = ring_participants[src_node_idx]
        dst_node = ring_participants[dst_node_idx]

        traversed_distance += distance_between_nodes(src_node, dst_node)

    return traversed_distance


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "-t",
        "--tp-degree",
        type=int,
        required=True,
        help="tensor parallel (TP) degree",
    )
    parser.add_argument(
        "-p",
        "--pp-degree",
        type=int,
        required=True,
        help="pipeline parallel (PP) degree",
    )
    parser.add_argument(
        "-d",
        "--dp-degree",
        type=int,
        required=True,
        help="data parallel (DP) degree",
    )
    parser.add_argument(
        "-n",
        "--accelerators-per-node",
        type=int,
        default=8,
        help="number of accelerators per node",
    )
    parser.add_argument(
        "-i",
        "--instance-type",
        type=str,
        help="required instance type",
    )
    parser.add_argument(
        "-r",
        "--region",
        type=str,
        help="required AWS region",
    )
    parser.add_argument(
        "-a",
        "--az",
        type=str,
        help="required AWS availability zone",
    )
    args = parser.parse_args()
    tp_degree = args.tp_degree
    pp_degree = args.pp_degree
    dp_degree = args.dp_degree
    accelerators_per_node = args.accelerators_per_node

    accepted_instance_ids = None  # TODO. maybe allow this to be read from a file
    instance_type = args.instance_type
    region_name = args.region
    az = args.az

    assert tp_degree <= accelerators_per_node

    # we only care to optimize inter-node DP communication
    # (intra-node DP communication can't be optimized with better instance placement),
    # so we'll ignore the parts of the ring that traverse intra-node links.
    dp_rank_per_node = safe_divide(accelerators_per_node, tp_degree)
    nodes_per_dp_ring = safe_divide(dp_degree, dp_rank_per_node)

    # everything we look at here is on the instance level.

    # if TP < accelerators per node, then there will be multiple DP ranks on a single
    # node. we'll ignore this intra-node DP communication as we just care about
    # improving inter-node DP communication
    # we're making the assumption that all TP groups are within a single node.
    # since we're looking at things at the instance level, there's only one
    # ring per PP rank.

    client = boto3.client("ec2", region_name=region_name)

    node_id_to_node: Dict[str, TreeNode] = {}

    # construct tree of instance topology
    instance_ids = set()
    for instance_info in dlcalc.utils.cluster.ec2.iter_instance_info(
        ec2_client=client,
        accepted_node_instance_types={
            instance_type,
        },
        accepted_node_availability_zones={
            az,
        },
        accepted_instance_ids=accepted_instance_ids,
    ):
        assert instance_info.instance_id not in node_id_to_node

        instance_ids.add(instance_info.instance_id)

        # order of network nodes: farthest from instance -> closest to instance
        chain = [*instance_info.network_nodes, instance_info.instance_id]
        for parent_node_idx in range(len(chain) - 1):
            src_node_id = chain[parent_node_idx]
            dst_node_id = chain[parent_node_idx + 1]

            for node_id in [src_node_id, dst_node_id]:
                if node_id not in node_id_to_node:
                    node_id_to_node[node_id] = TreeNode(node_id, parent=None, children=set())

            parent_node = node_id_to_node[src_node_id]
            child_node = node_id_to_node[dst_node_id]

            if child_node.parent is None:
                child_node.parent = parent_node
            elif child_node.parent != parent_node:
                raise AssertionError("mismatched parents")  # TODO. better error message

            parent_node.children.add(child_node)

    # find root node
    (root_node,) = [node for node in node_id_to_node.values() if node.parent is None]

    # determine optimal topology
    # we want to select an ordered subset of nodes for optimal adjacency.

    # the most adjacent set of nodes for ring collectives minimizes the number
    # of network nodes that need to be crossed.

    # Physical topology is a tree. in each level we can order by the subtree size,
    # then we'll walk it DFS. this will give us an ordered list of instances that would
    # make relatively suitable rings.
    # TODO. this is sort of a heuristic, will not always find the most optimal
    # placement.
    ordered_nodes = dfs_tree_leaves_only(root_node)

    if len(ordered_nodes) != len(instance_ids):
        raise RuntimeError(
            f"expected to traverse {len(instance_ids)} nodes but encountered {len(ordered_nodes)}"
        )

    # iterate PP ranks in reverse order, because we want to prioritize giving
    # the best rings to the final PP ranks, where DP communication is
    # hardest to overlap.
    ring_assignments: List[List[TreeNode]] = [None] * pp_degree
    for pp_rank in reversed(range(pp_degree)):
        ring_assignments[pp_rank] = ordered_nodes[:nodes_per_dp_ring]
        ordered_nodes = ordered_nodes[nodes_per_dp_ring:]

        print([n.node_id for n in ring_assignments[pp_rank]])
        print(traversal_distance_of_ring(ring_assignments[pp_rank]))
        print()


if __name__ == "__main__":
    main()
