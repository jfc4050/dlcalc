# dlcalc
![checks](https://github.com/jfc4050/dlcalc/actions/workflows/python-app.yml/badge.svg)

random command line tools for deep learning

install CLI
```bash
pip install -e .
```

## Tools
### 3D Training Calculator
calculator for estimating various performance characteristics of 3D parallel
transformer model training:
* memory consumption
* pipeline bubble
* communication overhead
* compute intensity
* etc..

```bash
3dtrn -h
```

we've include a sample config you can try tweaking
```bash
3dtrn examples/llama3_70b.yaml
```

### Checkpoint Summarizer
gives a human-readable summarization of keys, values, and tensor shapes in
a given training checkpoint.
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