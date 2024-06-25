"""
TODO.
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
    parser.add_argument("-s", "--search-prefix")
    parser.add_argument("-r", "--region")
    args = parser.parse_args()

    region = args.region
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

    node_id_to_pod_name = {}
    for node_info in client.list_node().items:
        # print(node_info)
        node_name = node_info.metadata.name
        node_id = json.loads(node_info.metadata.annotations["csi.volume.kubernetes.io/nodeid"])[
            "fsx.csi.aws.com"
        ]
        if node_name in node_name_to_pod_name:
            node_id_to_pod_name[node_id] = node_name_to_pod_name[node_name]

    if len(node_name_to_pod_name) != len(node_id_to_pod_name):
        raise RuntimeError  # TODO. fill out error message

    # worker_id = int(re.search(pattern, pod_name).group(1))

    instance_ids = list(node_id_to_pod_name.keys())
    print(instance_ids)

    ######################################################################################
    # STEP 2: Get topology of the instances acquired earlier.
    ######################################################################################
    ec2_client = boto3.client("ec2", region_name=region)

    paginator = ec2_client.get_paginator("describe_instance_topology")
    topo_info = []
    for page in paginator.paginate(InstanceIds=instance_ids):
        topo_info.extend(page["Instances"])

    print(topo_info)

    net = Network(directed=True)

    for instance_info in topo_info:
        instance_id = instance_info["InstanceId"]
        # instance_type = instance_info["InstanceType"]
        # order: farthest from instance -> closest to instance
        network_nodes = instance_info["NetworkNodes"]

        net.add_node(
            instance_id,
            title=_dict_to_label(
                {
                    "InstanceType": instance_info["InstanceType"],
                    "AvailabilityZone": instance_info["AvailabilityZone"],
                }
            ),
            color="blue",
            physics=False,
        )

        for network_node in network_nodes:
            net.add_node(network_node, color="red", physics=False)

        chain = [instance_id, *reversed(network_nodes)]

        for edge_src_idx in range(len(chain) - 1):
            edge_dst_idx = edge_src_idx + 1

            net.add_edge(chain[edge_src_idx], chain[edge_dst_idx], physics=False)

    net.show_buttons()
    net.show("graph.html", notebook=False)


if __name__ == "__main__":
    main()
