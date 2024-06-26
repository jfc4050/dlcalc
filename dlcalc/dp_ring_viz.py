"""
TODO.
"""

import dataclasses
import functools
from typing import Dict, List, Set

import boto3

import dlcalc.utils.cluster.ec2
from dlcalc.utils.math import safe_divide


@dataclasses.dataclass
class Node:
    node_id: str
    parent: "Node"
    children: Set["Node"]

    def __hash__(self) -> int:
        return hash(self.node_id)


def dfs_tree_leaves_only(root: Node) -> List[Node]:
    if not root.children:
        return [root]

    list_of_list_of_leaves = []
    for child in root.children:
        list_of_list_of_leaves.append(dfs_tree_leaves_only(child))

    # now we have a list of list of leaves acquired from each subtree.
    # we want this list of lists of leaves to be ordered from largest list -> smallest list
    list_of_list_of_leaves.sort(lambda l: len(l), reverse=True)

    list_of_leaves = sum(list_of_list_of_leaves)

    return list_of_leaves


@functools.lru_cache(maxsize=None)
def distance_between_nodes(node_a: Node, node_b: Node) -> int:
    if node_a == node_b:
        return 0

    return 2 + distance_between_nodes(node_a.parent, node_b.parent)


def traversal_distance_of_ring(ring_participants: List[Node]) -> int:
    traversed_distance = 0
    for src_node_idx in range(len(ring_participants)):
        dst_node_idx = src_node_idx + 1 % len(ring_participants)

        src_node = ring_participants[src_node_idx]
        dst_node = ring_participants[dst_node_idx]

        traversed_distance += distance_between_nodes(src_node, dst_node)

    return traversed_distance


def main() -> None:
    tp_degree = 2
    pp_degree = 1
    dp_degree = 80
    accelerators_per_node = 8

    accepted_instance_ids = None
    instance_type = "p5.48xlarge"
    region_name = "us-east-2"
    accepted_azs = {"us-east-2a",}

    assert tp_degree <= accelerators_per_node

    world_size = tp_degree * pp_degree * dp_degree
    n_nodes = safe_divide(world_size, accelerators_per_node)
    dp_rank_per_node = safe_divide(accelerators_per_node, tp_degree)
    instances_per_dp_ring = safe_divide(dp_degree, dp_rank_per_node)

    # everything we look at here is on the instance level.

    # if TP < accelerators per node, then there will be multiple DP ranks on a single
    # node. we'll ignore this intra-node DP communication as we just care about
    # improving inter-node DP communication
    # we're making the assumption that all TP groups are within a single node.
    # since we're looking at things at the instance level, there's only one
    # ring per PP rank.

    client = boto3.client("ec2", region_name=region_name)

    node_id_to_node: Dict[str, Node] = {}

    # construct tree of instance topology
    instance_ids = set()
    for instance_info in dlcalc.utils.cluster.ec2.iter_instance_info(
        ec2_client=client,
        accepted_node_instance_types={instance_type,},
        accepted_node_availability_zones=accepted_azs,
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
                    node_id_to_node[node_id] = Node(node_id, parent=None, children=set())

            parent_node = node_id_to_node[src_node_id]
            child_node = node_id_to_node[dst_node_id]

            if child_node.parent is None:
                child_node.parent = parent_node
            elif child_node.parent != parent_node:
                raise AssertionError("mismatched parents")  # TODO. better error message

            parent_node.children.add(dst_node_id)

    # find root node
    all_children = set().union(node.children for node in node_id_to_node.values())
    all_nodes = set(node for node in node_id_to_node.values())
    (root_node,) = all_nodes - all_children

    # determine optimal topology
    # we want to select an ordered subset of nodes for optimal adjacency.
    # we'll prioritize optimal adjacency for the final PP rank, as that is where
    # DP comm is hardest to overlap.

    # the most adjacent set of nodes for ring collectives minimizes the number
    # of network nodes that need to be crossed.

    # what is the path between a node and itself with size DP that crosses the fewest network links?
    # its a tree. in each level we can order by the subtree size, then we'll walk it DFS.
    # this will give us an ordered list of instances that would make relatively suitable rings.
    # then we can select rings out of this.

    ordered_nodes = dfs_tree_leaves_only(root_node)

    if len(ordered_nodes) != len(instance_ids):
        raise RuntimeError(
            f"expected to traverse {len(all_nodes)} nodes but encountered {len(ordered_nodes)}"
        )

    print([node.node_id for node in ordered_nodes])


if __name__ == "__main__":
    main()
