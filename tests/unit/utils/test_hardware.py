"""Unit tests for the hardware spec registry."""

import pytest

from dlcalc.utils.hardware import EFA_LATENCY_S, MachineSpec

GIB = 1024**3


def test_p5e_h200() -> None:
    spec = MachineSpec.from_str("p5e.48xlarge")

    assert spec.n_devices == 8
    assert spec.device_spec.peak_flops == int(989e12)
    assert spec.device_spec.mem_bandwidth_bytes_per_sec == int(4800e9)
    assert spec.device_spec.mem_capacity_bytes == 141 * GIB
    assert spec.device_spec.mamf == 0.859
    assert spec.intra_node_connect.unidirectional_bw_bytes_per_sec == int(450e9)
    # 3200 Gbps EFA shared by 8 GPUs
    assert spec.inter_node_connect.unidirectional_bw_bytes_per_sec == int(400e9 / 8)
    assert spec.inter_node_connect.latency_sec == EFA_LATENCY_S


def test_p6_b300() -> None:
    spec = MachineSpec.from_str("p6-b300.48xlarge")

    assert spec.n_devices == 8
    assert spec.device_spec.peak_flops == int(2250e12)
    assert spec.device_spec.mem_bandwidth_bytes_per_sec == int(8000e9)
    assert spec.device_spec.mem_capacity_bytes == 268 * GIB
    assert spec.device_spec.mamf == 0.841
    assert spec.intra_node_connect.unidirectional_bw_bytes_per_sec == int(900e9)
    # 6400 Gbps EFAv4 shared by 8 GPUs, double p6-b200
    assert spec.inter_node_connect.unidirectional_bw_bytes_per_sec == int(800e9 / 8)
    assert spec.inter_node_connect.latency_sec == EFA_LATENCY_S


def test_g7e_rtx_pro_6000() -> None:
    spec = MachineSpec.from_str("g7e.48xlarge")

    assert spec.n_devices == 8
    # NVIDIA lists 1 PFLOP BF16 with sparsity
    assert spec.device_spec.peak_flops == int(500e12)
    assert spec.device_spec.mem_bandwidth_bytes_per_sec == int(1597e9)
    assert spec.device_spec.mem_capacity_bytes == 96 * GIB
    # no NVLink, GPUs talk over PCIe Gen5 x16
    assert spec.intra_node_connect.unidirectional_bw_bytes_per_sec == int(64e9)
    # 1600 Gbps EFAv4 shared by 8 GPUs
    assert spec.inter_node_connect.unidirectional_bw_bytes_per_sec == int(200e9 / 8)
    assert spec.inter_node_connect.latency_sec == EFA_LATENCY_S


def test_trn2_logical_neuron_cores_as_devices() -> None:
    """trn2 is modeled as 64 logical NeuronCores (16 chips x 4 at the default LNC=2)."""
    spec = MachineSpec.from_str("trn2.48xlarge")

    assert spec.n_devices == 64
    assert spec.device_spec.peak_flops == int(10.7e15 / 64)
    assert spec.device_spec.mem_bandwidth_bytes_per_sec == int(46.4e12 / 64)
    assert spec.device_spec.mem_capacity_bytes == 24 * GIB
    # NeuronLink-v3 is 1024 GiB/s per chip, split across 4 logical cores
    assert spec.intra_node_connect.unidirectional_bw_bytes_per_sec == 256 * GIB
    # 3200 Gbps EFAv3 shared by 64 logical cores
    assert spec.inter_node_connect.unidirectional_bw_bytes_per_sec == int(400e9 / 64)
    assert spec.inter_node_connect.latency_sec == EFA_LATENCY_S


@pytest.mark.parametrize(
    "node_type", ["p5e.48xlarge", "p6-b300.48xlarge", "g7e.48xlarge", "trn2.48xlarge"]
)  # type: ignore[misc]
def test_total_flops(node_type: str) -> None:
    spec = MachineSpec.from_str(node_type)

    assert spec.total_flops() == spec.device_spec.peak_flops * spec.n_devices
