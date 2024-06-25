"""
given a kubernetes pod name prefix for some compute cluster, retrieve AWS network
topology info and plot.
"""

import json
import re
from argparse import ArgumentParser

import boto3
import kubernetes
from kubernetes.client import CoreV1Api
from pyvis.network import Network

pattern = re.compile(r"worker-(\d+)")


def _dict_to_label(dict: dict) -> str:
    kv_pairs = []
    for key, value in dict.items():
        kv_pairs.append(f"{key}: {value}")
    return "\n".join(kv_pairs)


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "search_prefix",
        help="kubernetes pod name prefix to use when filtering pods returned by `kubectl get pods`",
    )
    args = parser.parse_args()

    job_search_prefix = args.search_prefix

    ######################################################################################
    # STEP 1: Get the list of instance IDs involved in a kubernetes job
    ######################################################################################
    kubernetes.config.load_kube_config()
    client = CoreV1Api()

    node_name_to_pod_name = {}
    for pod_info in client.list_namespaced_pod(namespace="default", watch=False).items:
        pod_name: str = pod_info.metadata.name
        node_name: str = pod_info.spec.node_name

        if not pod_name.startswith(job_search_prefix):
            continue

        node_name_to_pod_name[node_name] = pod_name
    print(f"found {len(node_name_to_pod_name)} matching pods")

    node_regions = set()
    node_id_to_pod_name = {}
    for node_info in client.list_node().items:
        node_name = node_info.metadata.name
        node_id = json.loads(node_info.metadata.annotations["csi.volume.kubernetes.io/nodeid"])[
            "fsx.csi.aws.com"
        ]
        if node_name in node_name_to_pod_name:
            node_id_to_pod_name[node_id] = node_name_to_pod_name[node_name]

            node_region = node_info.metadata.labels["topology.kubernetes.io/region"]
            node_regions.add(node_region)

    (cluster_region,) = node_regions  # only handling single region clusters

    if len(node_name_to_pod_name) != len(node_id_to_pod_name):
        raise RuntimeError(
            f"found {len(node_name_to_pod_name)} pods but "
            f"was only able to match {len(node_id_to_pod_name)} to instance IDs"
        )

    instance_ids = list(node_id_to_pod_name.keys())

    ######################################################################################
    # STEP 2: Get topology of the instances acquired earlier.
    ######################################################################################
    print(f"getting topology info for instance IDs {instance_ids}")
    ec2_client = boto3.client("ec2", region_name=cluster_region)

    paginator = ec2_client.get_paginator("describe_instance_topology")
    topo_info = []
    for page in paginator.paginate(InstanceIds=instance_ids):
        topo_info.extend(page["Instances"])

    net = Network(directed=True)

    for instance_info in topo_info:
        instance_id = instance_info["InstanceId"]
        instance_type = instance_info["InstanceType"]
        network_nodes = instance_info["NetworkNodes"]

        pod_name = node_id_to_pod_name[instance_id]
        worker_id = int(re.search(pattern, pod_name).group(1))

        net.add_node(
            instance_id,
            title=_dict_to_label(
                {
                    "PodName": pod_name,
                    "WorkerId": worker_id,
                    "InstanceType": instance_type,
                    "AvailabilityZone": instance_info["AvailabilityZone"],
                }
            ),
            color="blue",
            physics=False,
        )

        for network_node in network_nodes:
            net.add_node(network_node, color="red", physics=False)

        # order of network nodes: farthest from instance -> closest to instance
        chain = [instance_id, *reversed(network_nodes)]
        for edge_src_idx in range(len(chain) - 1):
            edge_dst_idx = edge_src_idx + 1
            net.add_edge(chain[edge_src_idx], chain[edge_dst_idx], physics=False)

    net.show_buttons()
    net.show("graph.html", notebook=False)


if __name__ == "__main__":
    main()
