import json
from dataclasses import dataclass


@dataclass
class DeviceSpec:
    # TODO. make it depend on dtype
    peak_flops: int
    mem_bandwidth_bytes_per_sec: int
    mem_capacity_bytes: int


A100_40G_SPEC = DeviceSpec(
    peak_flops=int(312e12),
    mem_bandwidth_bytes_per_sec=int(1.55e12),
    mem_capacity_bytes=40 * (1024**3),
)
H100_SPEC = DeviceSpec(
    peak_flops=int(989e12),
    mem_bandwidth_bytes_per_sec=int(3.35e12),
    mem_capacity_bytes=80 * (1024**3),
)
NEURON_CORE_V2 = DeviceSpec(
    peak_flops=int(95e12),
    mem_bandwidth_bytes_per_sec=int(0.4e12),
    mem_capacity_bytes=16 * (1024**3),
)


@dataclass
class LinkSpec:
    unidirectional_bw_bits_per_sec: int
    latency_sec: float

    def __repr__(self) -> str:
        return json.dumps(
            {"unidirectional bw": f"{self.unidirectional_bw_bits_per_sec / 8 * 1e-9:.2f}GBps"}
        )


"""
NVLink:
v2, v3, v4 all have links w/ following specs:
* 6.25GBps/lane/direction
* 4 lanes

each version only differs by the number of links: v2 -> 6, v3 -> 12, v4 -> 18
"""
NVLINK2_SPEC = LinkSpec(
    unidirectional_bw_bits_per_sec=int(8 * 6.25 * 4 * 6 * 1e9),
    latency_sec=3e-6,
)
NVLINK3_SPEC = LinkSpec(
    unidirectional_bw_bits_per_sec=int(8 * 6.25 * 4 * 12 * 1e9),
    latency_sec=3e-6,
)
NVLINK4_SPEC = LinkSpec(
    unidirectional_bw_bits_per_sec=int(8 * 6.25 * 4 * 18 * 1e9),
    latency_sec=3e-6,
)

# TODO. double check this. not sure if public figure is duplex or not
NEURONLINK_V2_SPEC = LinkSpec(
    unidirectional_bw_bits_per_sec=348 * (1024**3),
    latency_sec=float("inf"),  # TODO. not sure, determine empirically.
)

EFA_LATENCY_S = 30e-6


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
                device_spec=A100_40G_SPEC,
                intra_node_connect=NVLINK3_SPEC,
                inter_node_connect=LinkSpec(
                    # spec page reports 400Gbps. assuming that's duplex.
                    unidirectional_bw_bits_per_sec=int(200e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # https://aws.amazon.com/ec2/instance-types/p5/
            "p5.48xlarge": MachineSpec(
                n_devices=8,
                device_spec=H100_SPEC,
                intra_node_connect=NVLINK4_SPEC,
                inter_node_connect=LinkSpec(
                    # spec page reports 3200Gbps. assuming that's duplex.
                    unidirectional_bw_bits_per_sec=int(1600e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # https://aws.amazon.com/ec2/instance-types/trn1/
            "trn1n.32xlarge": MachineSpec(
                n_devices=32,  # technically 16, but treating neuron cores as devices.
                device_spec=NEURON_CORE_V2,
                intra_node_connect=NEURONLINK_V2_SPEC,
                inter_node_connect=LinkSpec(
                    # spec page reports 1600Gbps. assuming that's duplex
                    unidirectional_bw_bits_per_sec=int(800e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
        }[str]

    def total_flops(self) -> int:
        return self.device_spec.peak_flops * self.n_devices
