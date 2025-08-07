"""Unit tests for model_3d module, focusing on ParallelConfig."""

import pytest

from dlcalc.utils.model_3d import ParallelConfig


class TestParallelConfig:
    """Test ParallelConfig class functionality."""

    def test_basic_initialization(self):
        """Test basic ParallelConfig initialization without expert mesh."""
        config = ParallelConfig(
            tp=4,
            cp=2,
            pp=2,
            dp=8,
            expert_mesh=None,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.tp == 4
        assert config.cp == 2
        assert config.pp == 2
        assert config.dp == 8
        assert config.expert_mesh is None
        assert config.vpp == 1
        assert config.sp_enabled is True
        assert config.zero_level == ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER

    def test_world_size_calculation(self):
        """Test world_size calculation."""
        config = ParallelConfig(
            tp=4,
            cp=2,
            pp=2,
            dp=8,
            expert_mesh=None,
            vpp=1,
            sp_enabled=False,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.world_size() == 4 * 2 * 2 * 8  # tp * cp * pp * dp = 128

    def test_expert_parallel_cfg_initialization(self):
        """Test ExpertParallelCfg initialization."""
        expert_cfg = ParallelConfig.ExpertParallelCfg(ep=8, tp=2, dp=4)

        assert expert_cfg.ep == 8
        assert expert_cfg.tp == 2
        assert expert_cfg.dp == 4

    def test_parallel_config_with_valid_expert_mesh(self):
        """Test ParallelConfig with a valid expert mesh configuration."""
        # The constraint is: expert_mesh.ep * expert_mesh.tp * expert_mesh.dp == dp * cp * tp
        # With tp=4, cp=2, dp=8: dp * cp * tp = 8 * 2 * 4 = 64
        # So expert_mesh.ep * expert_mesh.tp * expert_mesh.dp must equal 64
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=8,  # 8 * 2 * 4 = 64 ✓
            tp=2,
            dp=4,
        )

        config = ParallelConfig(
            tp=4,
            cp=2,
            pp=2,
            dp=8,
            expert_mesh=expert_mesh,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.expert_mesh.ep == 8
        assert config.expert_mesh.tp == 2
        assert config.expert_mesh.dp == 4

    def test_parallel_config_with_invalid_expert_mesh_raises_error(self):
        """Test that invalid expert mesh configuration raises assertion error."""
        # The constraint is: expert_mesh.ep * expert_mesh.tp * expert_mesh.dp == dp * cp * tp
        # With tp=4, cp=2, dp=8: dp * cp * tp = 8 * 2 * 4 = 64
        # But we'll provide expert_mesh that multiplies to 32 (invalid)
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=4,  # 4 * 2 * 4 = 32 ✗ (should be 64)
            tp=2,
            dp=4,
        )

        with pytest.raises(AssertionError):
            ParallelConfig(
                tp=4,
                cp=2,
                pp=2,
                dp=8,
                expert_mesh=expert_mesh,
                vpp=1,
                sp_enabled=True,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            )

    def test_zero_level_enum_values(self):
        """Test ZeroLevel enum values."""
        assert ParallelConfig.ZeroLevel.NONE == 0
        assert ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER == 1
        assert ParallelConfig.ZeroLevel.PARTITION_GRADIENTS == 2
        assert ParallelConfig.ZeroLevel.PARTITION_PARAMETERS == 3

    def test_different_world_sizes(self):
        """Test various world size configurations."""
        test_cases = [
            (1, 1, 1, 1, 1),  # Single device
            (2, 1, 1, 1, 2),  # TP only
            (1, 1, 2, 1, 2),  # PP only
            (1, 1, 1, 4, 4),  # DP only
            (2, 2, 2, 2, 16),  # All dimensions
            (8, 1, 4, 16, 512),  # Large cluster
        ]

        for tp, cp, pp, dp, expected_world_size in test_cases:
            config = ParallelConfig(
                tp=tp,
                cp=cp,
                pp=pp,
                dp=dp,
                expert_mesh=None,
                vpp=1,
                sp_enabled=False,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            )
            assert config.world_size() == expected_world_size

    def test_expert_mesh_constraint_with_cp_equals_1(self):
        """Test expert mesh constraint when cp=1."""
        # With tp=8, cp=1, dp=64: dp * cp * tp = 64 * 1 * 8 = 512
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=8,  # 8 * 8 * 8 = 512 ✓
            tp=8,
            dp=8,
        )

        config = ParallelConfig(
            tp=8,
            cp=1,
            pp=4,
            dp=64,
            expert_mesh=expert_mesh,
            vpp=1,
            sp_enabled=False,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.expert_mesh.ep == 8
        assert config.expert_mesh.tp == 8
        assert config.expert_mesh.dp == 8
        # Verify the constraint
        assert (
            config.expert_mesh.ep * config.expert_mesh.tp * config.expert_mesh.dp
            == config.dp * config.cp * config.tp
        )

    def test_expert_mesh_with_different_tp_values(self):
        """Test that expert_mesh.tp can differ from main tp."""
        # Main config has tp=4, but expert_mesh has tp=1
        # With tp=4, cp=2, dp=8: dp * cp * tp = 8 * 2 * 4 = 64
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=16,  # 16 * 1 * 4 = 64 ✓
            tp=1,  # Different from main tp=4
            dp=4,
        )

        config = ParallelConfig(
            tp=4,
            cp=2,
            pp=2,
            dp=8,
            expert_mesh=expert_mesh,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.tp == 4  # Main TP
        assert config.expert_mesh.tp == 1  # Expert TP (different)
        assert config.expert_mesh.ep == 16
        assert config.expert_mesh.dp == 4

    def test_vpp_values(self):
        """Test different VPP (Virtual Pipeline Parallel) values."""
        for vpp in [1, 2, 4, 8]:
            config = ParallelConfig(
                tp=2,
                cp=1,
                pp=4,
                dp=2,
                expert_mesh=None,
                vpp=vpp,
                sp_enabled=False,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            )
            assert config.vpp == vpp

    def test_sp_enabled_flag(self):
        """Test sequence parallel enablement flag."""
        config_sp_on = ParallelConfig(
            tp=2,
            cp=1,
            pp=2,
            dp=2,
            expert_mesh=None,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )
        assert config_sp_on.sp_enabled is True

        config_sp_off = ParallelConfig(
            tp=2,
            cp=1,
            pp=2,
            dp=2,
            expert_mesh=None,
            vpp=1,
            sp_enabled=False,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )
        assert config_sp_off.sp_enabled is False

    def test_edge_case_single_device_with_expert_mesh(self):
        """Test edge case with single device and expert mesh."""
        # With tp=1, cp=1, dp=1: dp * cp * tp = 1 * 1 * 1 = 1
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=1,  # 1 * 1 * 1 = 1 ✓
            tp=1,
            dp=1,
        )

        config = ParallelConfig(
            tp=1,
            cp=1,
            pp=1,
            dp=1,
            expert_mesh=expert_mesh,
            vpp=1,
            sp_enabled=False,
            zero_level=ParallelConfig.ZeroLevel.NONE,
        )

        assert config.world_size() == 1
        assert config.expert_mesh.ep == 1
        assert config.expert_mesh.tp == 1
        assert config.expert_mesh.dp == 1

    def test_large_scale_configuration(self):
        """Test large-scale cluster configuration."""
        # Simulating a large cluster with 12288 GPUs
        # tp=8, cp=4, pp=6, dp=64 => 8 * 4 * 6 * 64 = 12288
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=128,  # 128 * 8 * 2 = 2048 = dp * cp * tp = 64 * 4 * 8
            tp=8,
            dp=2,
        )

        config = ParallelConfig(
            tp=8,
            cp=4,
            pp=6,
            dp=64,
            expert_mesh=expert_mesh,
            vpp=2,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.world_size() == 12288
        assert config.expert_mesh.ep == 128
        # Verify constraint
        assert (
            config.expert_mesh.ep * config.expert_mesh.tp * config.expert_mesh.dp
            == config.dp * config.cp * config.tp
        )
