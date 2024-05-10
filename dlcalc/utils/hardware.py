import json
from dataclasses import dataclass


EFA_LATENCY_S = 30e-6


@dataclass
class DeviceSpec:
    # TODO. make it depend on dtype
    peak_flops: int
    mem_bandwidth_bytes_per_sec: int
    mem_capacity_bytes: int


@dataclass
class LinkSpec:
    unidirectional_bw_bits_per_sec: int
    latency_sec: float

    def __repr__(self) -> str:
        return json.dumps(
            {"unidirectional bw": f"{self.unidirectional_bw_bits_per_sec / 8 * 1e-9:.2f}GBps"}
        )


@dataclass
class MachineSpec:
    n_devices: int
    device_spec: DeviceSpec
    intra_node_connect: LinkSpec
    inter_node_connect: LinkSpec

    @staticmethod
    def from_str(str: str) -> "MachineSpec":
        return {
            # https://aws.amazon.com/ec2/instance-types/p4/
            "p4d.24xlarge": MachineSpec(
                n_devices=8,
                # A100-40G
                device_spec=DeviceSpec(
                    peak_flops=int(312e12),
                    # 40GiB HBM2
                    mem_bandwidth_bytes_per_sec=int(1555e9),
                    mem_capacity_bytes=40 * (1024**3),
                ),
                # NVLink 3
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bits_per_sec=int(2400e9),
                    latency_sec=3e-6,
                ),
                # EFA v1 - 4 x 100 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bits_per_sec=int(400e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # https://aws.amazon.com/ec2/instance-types/p4/
            "p4de.24xlarge": MachineSpec(
                n_devices=8,
                # A100-80G
                device_spec=DeviceSpec(
                    peak_flops=int(312e12),
                    # 80GiB HBM2e
                    mem_bandwidth_bytes_per_sec=int(1935e9),
                    mem_capacity_bytes=40 * (1024**3),
                ),
                # NVLink 3
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bits_per_sec=int(2400e9),
                    latency_sec=3e-6,
                ),
                # EFA v1 - 4 x 100 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bits_per_sec=int(400e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # https://aws.amazon.com/ec2/instance-types/p5/
            "p5.48xlarge": MachineSpec(
                n_devices=8,
                # H100
                device_spec=DeviceSpec(
                    peak_flops=int(989e12),
                    # 80 GiB HBM3
                    mem_bandwidth_bytes_per_sec=int(3350e9),
                    mem_capacity_bytes=80 * (1024**3),
                ),
                # NVLink 4
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bits_per_sec=int(3600e9),
                    latency_sec=3e-6,
                ),
                # EFA v2
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bits_per_sec=int(3200e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # https://aws.amazon.com/ec2/instance-types/trn1/
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn1-arch.html
            "trn1n.32xlarge": MachineSpec(
                n_devices=32,  # technically 16, but treating neuron cores as devices.
                # NeuronCore v2
                device_spec=DeviceSpec(
                    peak_flops=int(95e12),
                    mem_bandwidth_bytes_per_sec=int(400e9),
                    mem_capacity_bytes=16 * (1024**3),
                ),
                # NeuronLink v2
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bits_per_sec=int(412e9),
                    latency_sec=float("inf"),  # TODO. not sure, figure out empirically
                ),
                # EFA v2
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bits_per_sec=int(1600e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
        }[str]

    def total_flops(self) -> int:
        return self.device_spec.peak_flops * self.n_devices
