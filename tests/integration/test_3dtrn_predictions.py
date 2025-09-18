"""Integration tests for 3dtrn MFU predictions.

This module tests that the 3dtrn calculator produces expected MFU values
for known configurations.
"""

import re
import subprocess
from pathlib import Path

import pytest


def test_llama3_70b_mfu() -> None:
    _run_test("examples/llama3_70b.yaml", expected_mfu=30.22, expected_mem=31.965)


def test_gpt_oss_120b_mfu() -> None:
    _run_test("examples/gpt_oss_120b.yaml", expected_mfu=41.55, expected_mem=51.155)


@pytest.mark.parametrize(  # type: ignore[misc]
    "config_file",
    ["examples/llama3_70b.yaml", "examples/gpt_oss_120b.yaml"],
)
def test_output_format(config_file: str) -> None:
    """Test that 3dtrn output contains expected sections."""
    _, _, output = run_3dtrn(config_file)

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


def run_3dtrn(config_file: str, timeout: int = 30) -> tuple[float, float, str]:
    """Run 3dtrn and extract MFU and total memory values from output.

    Args:
        config_file: Path to YAML configuration file
        timeout: Maximum time to wait for command completion

    Returns:
        Tuple of (MFU percentage, total memory in GiB, full output)
    """
    result = subprocess.run(["3dtrn", config_file], capture_output=True, text=True, timeout=timeout)
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


TOLERANCE_PERC = 0.05


def _run_test(config_file: str, *, expected_mfu: float, expected_mem: float) -> None:
    assert Path(config_file).exists()

    actual_mfu, actual_mem, output = run_3dtrn(config_file)

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
