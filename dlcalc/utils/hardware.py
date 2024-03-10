from dataclasses import dataclass
import json


@dataclass
class DeviceSpec:
    # TODO. make it depend on dtype
    peak_tflops: int


A100_40G_SPEC = DeviceSpec(peak_tflops=312)


@dataclass
class LinkSpec:
    unidirectional_bw_bps: int

    def __repr__(self) -> str:
        return json.dumps({
            "unidirectional bw": f"{self.unidirectional_bw_bps / 8 * 1e-9:.2f}GBps"
        })

"""
NVLink:
v2, v3, v4 all have links w/ following specs:
* 6.25GBps/lane/direction
* 4 lanes

each version only differs by the number of links: v2 -> 6, v3 -> 12, v4 -> 18
"""
NVLINK2_SPEC = LinkSpec(unidirectional_bw_bps=8 * 6.25 * 4 * 6 * 1e9)
NVLINK3_SPEC = LinkSpec(unidirectional_bw_bps=8 * 6.25 * 4 * 12 * 1e9)
NVLINK4_SPEC = LinkSpec(unidirectional_bw_bps=8 * 6.25 * 4 * 18 * 1e9)

"""
EFA:
v1, v2 both use the same 100Gbps links,
only difference is that v2 has 32 while v1 has 4
"""
EFAV1_SPEC = LinkSpec(4 * 100 * 1e9)
EFAV2_SPEC = LinkSpec(32 * 100 * 1e9)


@dataclass
class MachineSpec:
    n_devices: int
    device_spec: DeviceSpec
    intra_node_connect: LinkSpec
    inter_node_connect: LinkSpec

    @staticmethod
    def from_str(str: str) -> "MachineSpec":
        return {
            "p4d.24xlarge": MachineSpec(
                n_devices=8,
                device_spec=A100_40G_SPEC,
                intra_node_connect=NVLINK3_SPEC,
                inter_node_connect=EFAV1_SPEC
            ),
        }[str]

    def total_flops(self) -> int:
        return self.device_spec.peak_tflops * self.n_devices
