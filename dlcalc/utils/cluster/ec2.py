from typing import Set

from botocore.client import BaseClient


def iter_instance_info(
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
