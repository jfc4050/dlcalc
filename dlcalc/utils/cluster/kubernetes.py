import dataclasses
import json
import re
from typing import List

import kubernetes
from kubernetes.client import CoreV1Api

pod_id_pattern = re.compile(r"worker-(\d+)")


@dataclasses.dataclass
class KubernetesJobMember:
    pod_name: str
    worker_id: int

    node_instance_type: str
    node_instance_id: str
    node_region: str
    node_az: str


def get_kubernetes_cluster_members(job_search_prefix: str) -> List[KubernetesJobMember]:
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

    cluster_members = []
    for node_info in client.list_node().items:
        node_name = node_info.metadata.name
        node_instance_id = json.loads(
            node_info.metadata.annotations["csi.volume.kubernetes.io/nodeid"]
        )["fsx.csi.aws.com"]
        if node_name in node_name_to_pod_name:
            pod_name = node_name_to_pod_name[node_name]
            worker_id = int(re.search(pod_id_pattern, pod_name).group(1))

            node_region = node_info.metadata.labels["topology.kubernetes.io/region"]
            cluster_regions.add(node_region)

            node_az = node_info.metadata.labels["topology.kubernetes.io/zone"]
            cluster_azs.add(node_az)

            node_instance_type = node_info.metadata.labels["node.kubernetes.io/instance-type"]
            cluster_instance_types.add(node_instance_type)

            member_def = KubernetesJobMember(
                pod_name=pod_name,
                worker_id=worker_id,
                node_instance_type=node_instance_type,
                node_instance_id=node_instance_id,
                node_region=node_region,
                node_az=node_az,
            )
            cluster_members.append(member_def)

    if len(cluster_regions) != 1:
        raise RuntimeError(f"found {len(cluster_regions)} regions but only expected 1.")

    if len(cluster_instance_types) != 1:
        print(f"WARNING: found {len(cluster_instance_types)}, generally there should only be 1.")

    if len(cluster_members) != len(node_name_to_pod_name):
        raise RuntimeError(
            f"found {len(node_name_to_pod_name)} pods but "
            f"was only able to match {len(cluster_members)} to instance IDs"
        )

    return cluster_members
