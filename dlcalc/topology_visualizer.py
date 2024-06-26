"""
given a kubernetes pod name prefix for some compute cluster, retrieve AWS network
topology info and plot.
"""

import json
import re
from argparse import ArgumentParser
from typing import Set

import boto3
import kubernetes
import tqdm
from botocore.client import BaseClient
from kubernetes.client import CoreV1Api
from pyvis.network import Network

pod_id_pattern = re.compile(r"worker-(\d+)")


def _iter_instance_info(
    ec2_client: BaseClient,
    accepted_instance_ids: Set[str],
    accepted_node_instance_types: Set[str],
    accepted_node_availability_zones: Set[str],
):
    # ref: https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-topology.html
    paginator = ec2_client.get_paginator("describe_instance_topology")

    # we have to filter in this roundabout way instead of giving a set of instance IDs
    # because API doesn't accept >100 instance IDs
    instance_filters = [
        {
            "Name": "instance-type",
            "Values": list(accepted_node_instance_types),
        },
        {
            "Name": "availability-zone",
            "Values": list(accepted_node_availability_zones),
        },
    ]

    for page in paginator.paginate(Filters=instance_filters, PaginationConfig={"PageSize": 100}):
        for instance_info in page["Instances"]:
            if instance_info["InstanceId"] in accepted_instance_ids:
                yield instance_info


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

    cluster_regions = set()
    cluster_azs = set()
    cluster_instance_types = set()

    node_id_to_pod_name = {}
    for node_info in client.list_node().items:
        node_name = node_info.metadata.name
        node_id = json.loads(node_info.metadata.annotations["csi.volume.kubernetes.io/nodeid"])[
            "fsx.csi.aws.com"
        ]
        if node_name in node_name_to_pod_name:
            node_id_to_pod_name[node_id] = node_name_to_pod_name[node_name]

            node_region = node_info.metadata.labels["topology.kubernetes.io/region"]
            cluster_regions.add(node_region)

            node_zone = node_info.metadata.labels["topology.kubernetes.io/zone"]
            cluster_azs.add(node_zone)

            node_instance_type = node_info.metadata.labels["node.kubernetes.io/instance-type"]
            cluster_instance_types.add(node_instance_type)

    (cluster_region,) = cluster_regions  # only handling single region clusters
    if len(cluster_instance_types) != 1:
        print(f"WARNING: found {len(cluster_instance_types)}, generally there should only be 1.")

    if len(node_name_to_pod_name) != len(node_id_to_pod_name):
        raise RuntimeError(
            f"found {len(node_name_to_pod_name)} pods but "
            f"was only able to match {len(node_id_to_pod_name)} to instance IDs"
        )

    instance_ids = set(node_id_to_pod_name.keys())

    ######################################################################################
    # STEP 2: Get topology of the instances acquired earlier.
    ######################################################################################
    print(f"getting topology info for instance IDs {instance_ids}")
    ec2_client = boto3.client("ec2", region_name=cluster_region)

    net = Network(directed=True)
    for instance_info in tqdm.tqdm(
        _iter_instance_info(
            ec2_client=ec2_client,
            accepted_instance_ids=instance_ids,
            accepted_node_instance_types=cluster_instance_types,
            accepted_node_availability_zones=cluster_azs,
        ),
        desc="processing instance info",
    ):
        instance_id = instance_info["InstanceId"]
        instance_type = instance_info["InstanceType"]
        network_nodes = instance_info["NetworkNodes"]

        pod_name = node_id_to_pod_name[instance_id]
        worker_id = int(re.search(pod_id_pattern, pod_name).group(1))

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
