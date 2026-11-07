import json
from dataclasses import dataclass

EFA_LATENCY_S = 30e-6


@dataclass
class DeviceSpec:
    # TODO. make it depend on dtype
    peak_flops: int
    mem_bandwidth_bytes_per_sec: int
    mem_capacity_bytes: int
    # Maximum Achievable Model FLOPS factor (0.0-1.0)
    # https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#maximum-achievable-matmul-flops-comparison-table
    mamf: float


@dataclass
class LinkSpec:
    unidirectional_bw_bytes_per_sec: int  # per GPU
    latency_sec: float

    def __repr__(self) -> str:
        return json.dumps(
            {
                "unidirectional bw/GPU": f"{self.unidirectional_bw_bytes_per_sec * 1e-9:.2f} GBps",
                "latency": f"{self.latency_sec * 1e6} us",
            }
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
            # A100 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
            # https://aws.amazon.com/ec2/instance-types/p4/
            "p4d.24xlarge": MachineSpec(
                n_devices=8,
                # A100-40GiB SXM
                device_spec=DeviceSpec(
                    peak_flops=int(312e12),
                    # 40GiB HBM2
                    mem_bandwidth_bytes_per_sec=int(1555e9),
                    mem_capacity_bytes=40 * (1024**3),
                    mamf=0.869,
                ),
                # NVLink 3
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(300e9),
                    latency_sec=3e-6,
                ),
                # EFA v1 - 4 x 100 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(50e9 / 8),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # https://aws.amazon.com/ec2/instance-types/p4/
            "p4de.24xlarge": MachineSpec(
                n_devices=8,
                # A100-80GiB SXM
                device_spec=DeviceSpec(
                    peak_flops=int(312e12),
                    # 80GiB HBM2e
                    mem_bandwidth_bytes_per_sec=int(2039e9),
                    mem_capacity_bytes=80 * (1024**3),
                    mamf=0.869,
                ),
                # NVLink 3
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(300e9),
                    latency_sec=3e-6,
                ),
                # EFA v1 - 4 x 100 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(50e9 / 8),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # H100 datasheet: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
            # https://aws.amazon.com/ec2/instance-types/p5/
            "p5.48xlarge": MachineSpec(
                n_devices=8,
                # H100 SXM
                device_spec=DeviceSpec(
                    peak_flops=int(989e12),  # (ignore sparsity numbers)
                    # 80 GiB HBM3
                    mem_bandwidth_bytes_per_sec=int(3350e9),
                    mem_capacity_bytes=80 * (1024**3),
                    mamf=0.803,
                ),
                # NVLink 4
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(450e9),
                    latency_sec=3e-6,
                ),
                # EFA v2 - 32 x 100 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(400e9 / 8),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # H200 datasheet: https://www.nvidia.com/en-us/data-center/h200/
            # https://aws.amazon.com/ec2/instance-types/p5/
            "p5e.48xlarge": MachineSpec(
                n_devices=8,
                # H200 SXM
                device_spec=DeviceSpec(
                    peak_flops=int(989e12),  # (ignore sparsity numbers)
                    # 141GB HBM3e
                    mem_bandwidth_bytes_per_sec=int(4800e9),
                    mem_capacity_bytes=141 * (1024**3),
                    mamf=0.859,
                ),
                # NVLink 4
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(450e9),
                    latency_sec=3e-6,
                ),
                # EFA v2 - 3200 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(400e9 / 8),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            "p6-b200.48xlarge": MachineSpec(
                n_devices=8,
                device_spec=DeviceSpec(
                    peak_flops=int(2250e12),
                    # 180GiB HBM3e
                    mem_bandwidth_bytes_per_sec=int(4916e9),
                    mem_capacity_bytes=180 * (1024**3),
                    mamf=0.776,
                ),
                # NVLink 5
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(900e9),
                    latency_sec=3e-6,  # TODO.
                ),
                # EFAv4
                # https://aws.amazon.com/ec2/instance-types/p6/
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(400e9 / 8),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # B300 bandwidth: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
            # HGX B300 (BF16 dense is 1/2 the listed sparse spec): https://www.nvidia.com/en-us/data-center/hgx/
            # https://aws.amazon.com/ec2/instance-types/p6/
            "p6-b300.48xlarge": MachineSpec(
                n_devices=8,
                device_spec=DeviceSpec(
                    peak_flops=int(2250e12),
                    # 288GB HBM3e, 2144GB per instance visible on AWS
                    mem_bandwidth_bytes_per_sec=int(8000e9),
                    mem_capacity_bytes=268 * (1024**3),
                    mamf=0.841,
                ),
                # NVLink 5
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(900e9),
                    latency_sec=3e-6,  # TODO.
                ),
                # EFAv4 - 6400 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(800e9 / 8),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            "u-p6e-gb300x72": MachineSpec(
                n_devices=72,
                device_spec=DeviceSpec(
                    peak_flops=int(2500e12),
                    mem_bandwidth_bytes_per_sec=int(8192e9),
                    mem_capacity_bytes=284 * (1024**3),
                    mamf=0.729,  # don't have GB300 values, using GB200 MAMF for now
                ),
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(1848e9),
                    latency_sec=3e-6,  # TODO.
                ),
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(800e9 / 8),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # RTX PRO 6000 datasheet: https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
            # https://aws.amazon.com/ec2/instance-types/g7e/
            "g7e.48xlarge": MachineSpec(
                n_devices=8,
                # RTX PRO 6000 Blackwell Server Edition
                device_spec=DeviceSpec(
                    peak_flops=int(500e12),  # 1 PFLOP listed is with sparsity
                    # 96GB GDDR7
                    mem_bandwidth_bytes_per_sec=int(1597e9),
                    mem_capacity_bytes=96 * (1024**3),
                    mamf=0.7,  # TODO. not in MAMF tables, measure empirically
                ),
                # no NVLink, GPUDirect P2P over PCIe Gen5 x16
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(64e9),
                    latency_sec=3e-6,  # TODO. placeholder copied from NVLink
                ),
                # EFAv4 - 1600 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(200e9 / 8),
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
                    # 16 GiB HBM2e
                    mem_bandwidth_bytes_per_sec=int(410e9),
                    mem_capacity_bytes=16 * (1024**3),
                    mamf=0.7,  # TODO. not sure, figure out empirically
                ),
                # NeuronLink v2
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(384e9),
                    latency_sec=float("inf"),  # TODO. not sure, figure out empirically
                ),
                # EFA v2
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(200e9 / 32),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html
            "trn2.48xlarge": MachineSpec(
                # 16 Trainium2 chips x 4 logical NeuronCores at the default LNC=2.
                # Like trn1n, treating (logical) neuron cores as devices.
                n_devices=64,
                device_spec=DeviceSpec(
                    # 10.7 PFLOPS BF16 dense per instance
                    peak_flops=int(10.7e15 / 64),
                    # 96 GiB HBM per chip, 46.4 TB/s per instance
                    mem_bandwidth_bytes_per_sec=int(46.4e12 / 64),
                    mem_capacity_bytes=24 * (1024**3),
                    mamf=0.7,  # TODO. not sure, figure out empirically
                ),
                # NeuronLink v3 - 1024 GiB/s per chip
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=256 * (1024**3),
                    # TODO. placeholder copied from NVLink, figure out empirically
                    latency_sec=3e-6,
                ),
                # EFA v3 - 3200 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(400e9 / 64),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
        }[str]

    def total_flops(self) -> int:
        return self.device_spec.peak_flops * self.n_devices
