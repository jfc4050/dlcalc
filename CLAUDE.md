# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

dlcalc is a Python CLI toolkit for deep learning performance calculations and optimization, focused on transformer model training with 3D parallelism (data, tensor, and pipeline parallelism). The package provides command-line tools for performance modeling, topology optimization, and training metrics calculations.

## Development Commands

### Static Analysis and Code Quality
Run all checks (formatting, linting, type checking, tests):
```bash
uv run bash checks
```

This runs:
- `ruff format --diff .` - Check formatting without modifying files
- `ruff check --output-format=github .` - Lint code
- `mypy dlcalc/training_3d.py` - Type check the main module
- `pytest tests/ -v --cov=dlcalc --cov-report=term-missing` - Run tests with coverage

To auto-format code:
```bash
uv run ruff format .
```

### Pre-commit Hooks
Install pre-commit hooks for automatic code quality checks:
```bash
pre-commit install
```

Run pre-commit manually on all files:
```bash
pre-commit run --all-files
```

### Installation and Dependency Management

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

#### Setting up the development environment
```bash
# Install all dependencies including dev extras
uv sync --all-extras
```

#### Adding new dependencies
```bash
# Add a production dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add with specific version constraints
uv add "package-name>=1.0.0"
```

#### Updating dependencies
```bash
# Update all dependencies to latest compatible versions
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package package-name

# Apply the updates
uv sync
```

#### Running commands without activating venv
```bash
# Run any command in the uv environment
uv run python dlcalc/training_3d.py configs/example.yaml

# Run tests
uv run pytest tests/

# Run the checks script
uv run bash checks
```

### Testing
Run the test suite with coverage:
```bash
uv run pytest tests/ -v --cov=dlcalc --cov-report=term-missing
```

Run specific test files:
```bash
uv run pytest tests/unit/calculators/test_memory.py -v
```

Run tests matching a pattern:
```bash
uv run pytest tests/ -k "test_memory" -v
```

## Architecture and Key Components

### Core Modules Structure
- **dlcalc/training_3d.py**: Main 3D parallelism calculator implementing performance modeling equations for transformer training. Contains the core logic for memory, compute, and communication calculations.
- **dlcalc/calculators/**: Refactored pure calculation functions for improved testability
  - `memory.py`: Memory requirement calculations (weights, optimizer, gradients, activations)
  - `flops.py`: FLOP calculations for forward/backward passes
  - `communication.py`: Communication overhead and pipeline bubble calculations
- **dlcalc/hardware_registry.py**: Configurable hardware specification management with dependency injection support
- **dlcalc/topoviz.py**: Kubernetes topology visualization using pyvis for network graph generation
- **dlcalc/topoeval.py**: Evaluates network topologies for optimal training job placement
- **dlcalc/topoassign.py**: Implements topology-aware scheduling algorithms for distributed training
- **dlcalc/utils/**: Shared utilities for parameter parsing and common calculations

### CLI Entry Points
All CLI commands are defined in `pyproject.toml` under `[project.scripts]`:
- `3dtrn`: Performance calculator for 3D parallel training
- `topoviz`, `topoeval`, `topoassign`: Topology optimization tools
- `sps2tpd`, `sps2mfu`: Metric conversion utilities
- `ckpt-summarize`: PyTorch checkpoint analysis

### Configuration System
- Hardware specifications stored in `configs/` as YAML files (e.g., `configs/a100_40gb.yaml`)
- Example training configurations in `examples/` directory
- Configuration loading handled through PyYAML with custom parameter parsing in utils module

### Key Design Patterns
1. Each CLI tool is a standalone module with a `main()` function
2. Hardware specs and model configs are externalized to YAML files
3. The 3D training calculator uses analytical equations for performance modeling (no simulation)
4. Topology tools integrate with Kubernetes API for real cluster analysis

## Important Implementation Details

### Type Checking
The project uses strict MyPy configuration. When adding new code:
- Add type hints to all function signatures
- Use `Optional[]` for nullable types
- Run `mypy dlcalc/training_3d.py` to verify types

### Code Style
- Line length: 100 characters (enforced by Ruff)
- Import sorting: Handled automatically by Ruff's isort rules
- Use Ruff for both formatting and linting

### AWS and Kubernetes Integration
- AWS operations use boto3 client
- Kubernetes operations use the official Python client
- Topology tools expect proper kubeconfig for cluster access

### Performance Calculations
The 3D training calculator implements equations from research papers on distributed training. Key concepts:
- Pipeline bubble analysis for pipeline parallelism efficiency
- All-reduce and point-to-point communication modeling
- Memory consumption across different parallelism dimensions
- MFU (Model FLOPs Utilization) calculations based on hardware specs