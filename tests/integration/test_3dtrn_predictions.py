"""Integration tests for 3dtrn MFU predictions.

This module tests that the 3dtrn calculator produces expected MFU values
for known configurations.
"""

import re
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml


@pytest.mark.parametrize(  # type: ignore[misc]
    "config_file,expected_mfu,expected_mem",
    [
        ("examples/llama3_70b.yaml", 30.24, 31.965),
        ("examples/llama3_70b_cross_dc.yaml", 29.61, 31.404),
        ("examples/gpt_oss_120b.yaml", 30.89, 51.155),
    ],
)
def test_output_format(config_file: str, expected_mfu: float, expected_mem: float) -> None:
    """Test that 3dtrn output contains expected sections."""
    TOLERANCE_PERC = 1e-3

    assert Path(config_file).exists()

    actual_mfu, actual_mem, output = _run_3dtrn(config_file)

    # check MFU value with tolerance
    mfu_diff = abs(actual_mfu - expected_mfu) / expected_mfu
    assert mfu_diff <= TOLERANCE_PERC, (
        f"MFU prediction {actual_mfu}% is outside tolerance. Expected: {expected_mfu}%"
    )

    # check memory value with tolerance
    mem_diff = abs(actual_mem - expected_mem) / expected_mem
    assert mem_diff <= TOLERANCE_PERC, (
        f"Mem prediction {actual_mem}GiB is outside tolerance. Expected: {expected_mem}GiB"
    )

    # Check for expected sections in output (remove ANSI codes first)
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    expected_sections = [
        "CONFIGURATION",
        "Model Architecture",
        "MEMORY",
        "Theoretical MFU:",
    ]

    missing_sections = []
    for section in expected_sections:
        if section not in clean_output:
            missing_sections.append(section)

    assert not missing_sections, (
        f"Output missing expected sections: {missing_sections}\nOutput sample:\n{output[:500]}"
    )


def test_missing_file() -> None:
    """Test that 3dtrn handles missing files gracefully."""
    result = subprocess.run(["3dtrn", "non_existent_file.yaml"], capture_output=True, text=True)

    assert result.returncode != 0, "Should return non-zero exit code for missing file"

    error_output = result.stdout + result.stderr
    assert (
        "No such file" in error_output
        or "not found" in error_output
        or "does not exist" in error_output
    ), f"Should report file not found error, got: {error_output}"


def test_invalid_yaml(tmp_path: Path) -> None:
    """Test that 3dtrn handles invalid YAML gracefully."""
    # Create invalid YAML file
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("{ this is: invalid yaml ][")

    result = subprocess.run(["3dtrn", str(invalid_yaml)], capture_output=True, text=True)

    assert result.returncode != 0, "Should return non-zero exit code for invalid YAML"

    error_output = result.stdout + result.stderr
    assert "yaml" in error_output.lower() or "parse" in error_output.lower(), (
        f"Should report YAML parsing error, got: {error_output}"
    )


def test_missing_required_fields(tmp_path: Path) -> None:
    """Test that 3dtrn handles configs with missing required fields."""
    # Create YAML with missing fields
    incomplete_config = tmp_path / "incomplete.yaml"
    incomplete_config.write_text("""
model:
n_layers: 32
# Missing other required fields
""")
    result = subprocess.run(["3dtrn", str(incomplete_config)], capture_output=True, text=True)

    assert result.returncode != 0, "Should return non-zero exit code for incomplete config"


def test_cross_dc_config_parsing() -> None:
    """Test that 3dtrn correctly parses and uses cross_dc configuration."""
    # Create a test config with cross-DC parameters
    config = {
        "model": {
            "n_layers": 32,
            "hidden_sz": 4096,
            "inter_sz": 11008,
            "n_q_heads": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "vocab_sz": 32000,
            "glu": True,
            "rotary_embeds": True,
            "dropout": False,
            "tie_embeddings": True,
        },
        "parallelism": {
            "tp": 8,
            "pp": 2,
            "dp": 16,
            "vpp": 1,
            "sp": True,
            "zero_level": 1,
            "n_param_buckets": 5,
        },
        "performance": {
            "activation_checkpointing_type": "selective",
        },
        "data": {
            "gbs": 256,
            "seqlen": 2048,
            "microbatch_sz": 1,
        },
        "hardware": {
            "node_type": "p5.48xlarge",
        },
        "cross_dc": {
            "n_dcs": 3,
            "interconnect_bandwidth_gbps": 800,
            "interconnect_latency_s": 0.0035,
        },
    }

    # Write config to temporary file
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        # Run 3dtrn with cross-DC config
        mfu, memory, output = _run_3dtrn(config_path)

        # Check that output contains cross-DC information
        assert "cross-DC" in output or "Cross-DC" in output

        # MFU should be lower than baseline due to cross-DC overhead
        # (We can't check exact values without running baseline, but it should be reasonable)
        assert mfu is not None
        assert 0 < mfu < 100  # MFU should be between 0 and 100%

    finally:
        # Clean up temporary file
        Path(config_path).unlink()


def test_cross_dc_degradation_reporting() -> None:
    """Test that cross-DC configuration shows degradation in output."""
    # Configuration with cross-DC
    config = {
        "model": {
            "n_layers": 32,
            "hidden_sz": 4096,
            "inter_sz": 11008,
            "n_q_heads": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "vocab_sz": 32000,
            "glu": True,
            "rotary_embeds": True,
            "dropout": False,
            "tie_embeddings": True,
        },
        "parallelism": {
            "tp": 8,
            "pp": 2,
            "dp": 16,
            "vpp": 1,
            "sp": True,
            "zero_level": 1,
            "n_param_buckets": 5,
        },
        "performance": {
            "activation_checkpointing_type": "selective",
        },
        "data": {
            "gbs": 256,
            "seqlen": 2048,
            "microbatch_sz": 1,
        },
        "hardware": {
            "node_type": "p5.48xlarge",
        },
        "cross_dc": {
            "n_dcs": 3,
            "interconnect_bandwidth_gbps": 800,
            "interconnect_latency_s": 0.0035,
        },
    }

    # Run with cross-DC
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        mfu, _, output = _run_3dtrn(config_path)
    finally:
        Path(config_path).unlink()

    # Verify cross-DC impact is reported
    assert "Cross-DC Impact" in output or "cross-DC" in output
    assert "degradation" in output.lower() or "slower" in output.lower()

    # Should show specific cross-DC metrics
    assert "Cross-DC Configuration" in output or "cross_dc" in output
    assert "800" in output  # bandwidth
    assert "3.5" in output or "0.0035" in output  # latency


def test_cross_dc_output_format() -> None:
    """Test that cross-DC output includes expected information."""
    config = {
        "model": {
            "n_layers": 16,
            "hidden_sz": 2048,
            "inter_sz": 5504,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "head_dim": 128,
            "vocab_sz": 32000,
            "glu": True,
            "rotary_embeds": True,
            "dropout": False,
            "tie_embeddings": True,
        },
        "parallelism": {
            "tp": 4,
            "pp": 2,
            "dp": 8,
            "vpp": 1,
            "sp": True,
            "zero_level": 1,
            "n_param_buckets": 5,
        },
        "performance": {
            "activation_checkpointing_type": "selective",
        },
        "data": {
            "gbs": 64,
            "seqlen": 1024,
            "microbatch_sz": 1,
        },
        "hardware": {
            "node_type": "p4d.24xlarge",
        },
        "cross_dc": {
            "n_dcs": 2,
            "interconnect_bandwidth_gbps": 400,
            "interconnect_latency_s": 0.005,
        },
    }

    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        mfu, memory, output = _run_3dtrn(config_path)

        # Check for expected cross-DC output elements
        assert "Cross-DC Configuration" in output or "cross_dc" in output
        assert "degradation" in output.lower()

        # Should show bandwidth and latency information
        assert "800" in output or "400" in output  # bandwidth
        assert "0.005" in output or "5" in output  # latency

    finally:
        Path(config_path).unlink()


def _run_3dtrn(config_file: str) -> tuple[float, float, str]:
    """Run 3dtrn and extract MFU and total memory values from output.

    Args:
        config_file: Path to YAML configuration file
        timeout: Maximum time to wait for command completion

    Returns:
        Tuple of (MFU percentage, total memory in GiB, full output)
    """
    result = subprocess.run(["3dtrn", config_file], capture_output=True, text=True, timeout=5)
    output = result.stdout + result.stderr

    # Find MFU in output - looking for "Theoretical MFU: XX.XX%" (with potential ANSI codes)
    # Remove ANSI escape codes first for easier matching
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    mfu_match = re.search(r"Theoretical MFU:\s*([0-9]+\.[0-9]+)%", clean_output)
    if not mfu_match:
        raise RuntimeError(f"Could not find MFU in output:\n{clean_output}")
    mfu_value = float(mfu_match.group(1))

    # Find total memory in output - looking for "Total Memory Required: XX.XXX GiB"
    mem_match = re.search(r"Total Memory Required:\s*([0-9]+\.[0-9]+)\s*GiB", clean_output)
    if not mem_match:
        raise RuntimeError(f"Could not find total memory in output:\n{clean_output}")
    mem_value = float(mem_match.group(1))

    return mfu_value, mem_value, output
