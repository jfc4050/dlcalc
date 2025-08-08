"""Integration tests for 3dtrn MFU predictions.

This module tests that the 3dtrn calculator produces expected MFU values
for known configurations.
"""

import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pytest


def run_3dtrn(config_file: str, timeout: int = 30) -> Tuple[Optional[float], str]:
    """Run 3dtrn and extract MFU value from output.

    Args:
        config_file: Path to YAML configuration file
        timeout: Maximum time to wait for command completion

    Returns:
        Tuple of (MFU percentage or None if not found, full output)
    """
    try:
        result = subprocess.run(
            ["3dtrn", config_file], capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr

        # Find MFU in output - looking for "predicted MFU: XX.XX%"
        mfu_match = re.search(r"predicted MFU:\s*([0-9]+\.[0-9]+)%", output)
        if mfu_match:
            return float(mfu_match.group(1)), output

        return None, output

    except subprocess.TimeoutExpired:
        pytest.fail(f"Timeout ({timeout}s) running {config_file}")
    except FileNotFoundError:
        pytest.skip("3dtrn command not found - install package first")
    except Exception as e:
        pytest.fail(f"Error running {config_file}: {e}")


class TestMFUPredictions:
    """Test MFU predictions for various model configurations."""

    def test_llama3_70b_mfu(self):
        """Test that LLaMA-3 70B configuration predicts ~30.22% MFU."""
        config_file = "examples/llama3_70b.yaml"

        # Skip if config file doesn't exist
        if not Path(config_file).exists():
            pytest.skip(f"Config file {config_file} not found")

        # Run 3dtrn and get MFU
        actual_mfu, output = run_3dtrn(config_file)

        # Check that we got an MFU value
        assert actual_mfu is not None, f"Could not find MFU in output:\n{output[-1000:]}"

        # Expected value with tolerance
        expected_mfu = 30.22
        tolerance = 0.5  # Allow 0.5% difference

        # Check if within tolerance
        difference = abs(actual_mfu - expected_mfu)
        assert difference <= tolerance, (
            f"MFU prediction {actual_mfu}% is outside tolerance. "
            f"Expected: {expected_mfu}% ± {tolerance}%, "
            f"Difference: {difference:.2f}%"
        )

    def test_gpt_oss_120b_mfu(self):
        """Test that GPT OSS 120B MoE configuration predicts the right MFU."""
        config_file = "examples/gpt_oss_120b.yaml"

        # Skip if config file doesn't exist
        if not Path(config_file).exists():
            pytest.skip(f"Config file {config_file} not found")

        # Run 3dtrn and get MFU
        actual_mfu, output = run_3dtrn(config_file)

        # Check that we got an MFU value
        assert actual_mfu is not None, f"Could not find MFU in output:\n{output[-1000:]}"

        # Expected value with tolerance
        expected_mfu = 17.3
        tolerance = 0.5  # Allow 0.5% difference

        # Check if within tolerance
        difference = abs(actual_mfu - expected_mfu)
        assert difference <= tolerance, (
            f"MFU prediction {actual_mfu}% is outside tolerance. "
            f"Expected: {expected_mfu}% ± {tolerance}%, "
            f"Difference: {difference:.2f}%"
        )

        # Check that output mentions MoE since this is an MoE model
        assert "MoE" in output or "expert" in output.lower(), (
            "Output should mention MoE configuration"
        )

    def test_multiple_configs(self):
        """Test MFU predictions for multiple configurations if available."""
        # Define test cases: (config_file, expected_mfu, tolerance)
        test_cases = [
            ("examples/llama3_70b.yaml", 30.22, 0.5),
            ("examples/gpt_oss_120b.yaml", 17.3, 0.5),
            # Add more test cases as needed
            # ("examples/llama3_8b.yaml", 45.0, 1.0),
        ]

        results = []
        for config_file, expected_mfu, tolerance in test_cases:
            if not Path(config_file).exists():
                results.append((config_file, "SKIPPED", "File not found"))
                continue

            actual_mfu, output = run_3dtrn(config_file)

            if actual_mfu is None:
                results.append((config_file, "FAILED", "MFU not found in output"))
                continue

            difference = abs(actual_mfu - expected_mfu)
            if difference <= tolerance:
                results.append((config_file, "PASSED", f"{actual_mfu}%"))
            else:
                results.append(
                    (
                        config_file,
                        "FAILED",
                        f"{actual_mfu}% (expected {expected_mfu}% ± {tolerance}%)",
                    )
                )

        # Print summary
        print("\nMFU Prediction Test Results:")
        print("-" * 60)
        for config, status, details in results:
            print(f"{config:40} {status:8} {details}")

        # Check if any failed
        failed = [r for r in results if r[1] == "FAILED"]
        if failed:
            pytest.fail(f"{len(failed)} configuration(s) failed MFU prediction test")

    def test_output_format(self):
        """Test that 3dtrn output contains expected sections."""
        config_file = "examples/llama3_70b.yaml"

        if not Path(config_file).exists():
            pytest.skip(f"Config file {config_file} not found")

        _, output = run_3dtrn(config_file)

        # Check for expected sections in output
        expected_sections = [
            "CONFIG",
            "MODEL SUMMARY",
            "[MEMORY]",
            "predicted MFU:",
        ]

        missing_sections = []
        for section in expected_sections:
            if section not in output:
                missing_sections.append(section)

        assert not missing_sections, (
            f"Output missing expected sections: {missing_sections}\nOutput sample:\n{output[:500]}"
        )

    @pytest.mark.parametrize(
        "batch_size,expected_range",
        [
            (2048, (25, 35)),  # Normal batch size
            (1024, (20, 30)),  # Smaller batch
            (4096, (30, 40)),  # Larger batch
        ],
    )
    def test_batch_size_impact(self, batch_size, expected_range, tmp_path):
        """Test that batch size affects MFU as expected."""
        # Create a temporary config with different batch sizes
        base_config = Path("examples/llama3_70b.yaml")

        if not base_config.exists():
            pytest.skip("Base config file not found")

        import yaml

        # Load base config
        with open(base_config) as f:
            config = yaml.safe_load(f)

        # Modify batch size
        config["data"]["gbs"] = batch_size

        # Save to temp file
        temp_config = tmp_path / f"test_bs_{batch_size}.yaml"
        with open(temp_config, "w") as f:
            yaml.dump(config, f)

        # Run and check MFU
        actual_mfu, output = run_3dtrn(str(temp_config))

        if actual_mfu is None:
            pytest.fail(f"Could not extract MFU for batch size {batch_size}")

        # Check if in expected range
        min_mfu, max_mfu = expected_range
        assert min_mfu <= actual_mfu <= max_mfu, (
            f"MFU {actual_mfu}% for batch size {batch_size} "
            f"outside expected range [{min_mfu}, {max_mfu}]"
        )


class TestErrorHandling:
    """Test error handling in 3dtrn."""

    def test_missing_file(self):
        """Test that 3dtrn handles missing files gracefully."""
        result = subprocess.run(["3dtrn", "non_existent_file.yaml"], capture_output=True, text=True)

        assert result.returncode != 0, "Should return non-zero exit code for missing file"

        error_output = result.stdout + result.stderr
        assert (
            "No such file" in error_output
            or "not found" in error_output
            or "does not exist" in error_output
        ), f"Should report file not found error, got: {error_output}"

    def test_invalid_yaml(self, tmp_path):
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

    def test_missing_required_fields(self, tmp_path):
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


@pytest.mark.slow
class TestPerformance:
    """Performance tests for 3dtrn (marked as slow)."""

    def test_execution_time(self):
        """Test that 3dtrn completes within reasonable time."""
        import time

        config_file = "examples/llama3_70b.yaml"

        if not Path(config_file).exists():
            pytest.skip(f"Config file {config_file} not found")

        start_time = time.time()
        run_3dtrn(config_file, timeout=10)
        execution_time = time.time() - start_time

        # Should complete within 5 seconds for a single config
        assert execution_time < 5, f"Execution took {execution_time:.2f}s, expected < 5s"


# Fixtures for common test data
@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration for testing."""
    config = {
        "model": {
            "n_layers": 32,
            "hidden_sz": 4096,
            "inter_sz": 11008,
            "n_q_heads": 32,
            "n_kv_heads": 32,
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
            "dp": 16,
            "vpp": 1,
            "sp": True,
            "zero_level": 1,
            "bucket_size_mb": 250,
        },
        "performance": {"activation_checkpointing_type": "selective"},
        "data": {"gbs": 512, "seqlen": 2048, "microbatch_sz": 1},
        "hardware": {"node_type": "p4d.24xlarge"},
    }

    config_file = tmp_path / "sample_config.yaml"
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file


def test_with_sample_config(sample_config):
    """Test 3dtrn with a generated sample configuration."""
    actual_mfu, output = run_3dtrn(str(sample_config))

    # Should produce some MFU value
    assert actual_mfu is not None, "Should calculate MFU for sample config"
    assert 0 < actual_mfu < 100, f"MFU should be between 0 and 100, got {actual_mfu}"

    # Should have standard output sections
    assert "CONFIG" in output
    assert "MODEL SUMMARY" in output
    assert "[MEMORY]" in output
