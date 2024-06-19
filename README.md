# dlcalc
[![PyPI version](https://badge.fury.io/py/dlcalc.svg)](https://badge.fury.io/py/dlcalc)
![checks](https://github.com/jfc4050/dlcalc/actions/workflows/python-app.yml/badge.svg)

random command line tools for deep learning

## Installation
```bash
pip install dlcalc
```

or

```bash
git clone https://github.com/jfc4050/dlcalc
cd dlcalc
pip install -e .
```

After this you should have access to the command line tools described below. Some
people may need to add `--user` to their pip install command for them to properly
go under `$PATH`.

## Tools
### 3D Training Calculator
Calculator for estimating various performance characteristics of 3D parallel
transformer model training:
* memory consumption
* pipeline bubble
* communication overhead
* compute intensity
* etc..

For more details run:
```bash
3dtrn -h
```

We've include a sample config you can use to see what the output looks like:
```bash
3dtrn examples/llama3_70b.yaml
```

We recommend pairing this with a profiler of your choice
([NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) and
[PyTorch Profiler Traces](https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_chrome_trace)
are two good ones), and checking that what you see in the profiler is in the same
ballpark as the theoretical values estimated by the calculator.

### Samples/Sec -> Tokens/Day Converter
Pretty self explanatory, for more details run:
```bash
sps2tpd -h
```

### Samples/Sec -> MFU Converter
If you're not familiar with what Model Flops Utilization (MFU) means, refer to
Google's [PaLM paper](https://arxiv.org/pdf/2204.02311). Otherwise pretty self
explanatory, for more details run:
```bash
sps2mfu -h
```

### Checkpoint Summarizer
Gives a human-readable summarization of keys, values, and tensor shapes in
a given (PyTorch) model checkpoint. For more details run:
```bash
ckpt-summarize -h
```

## Development
install development dependencies
```bash
pip install -e .[dev]
```

static checks can be run with
```bash
bash checks
```