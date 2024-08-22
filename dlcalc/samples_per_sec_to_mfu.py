"""simple utility to convert samples/second to MFU."""

import re
from argparse import ArgumentParser

n_params_pattern = re.compile(r"([\d\.]+)([a-z])")
n_accelerators_pattern = re.compile(r"(\d+)x(\d+)")


def parse_n_params(n_params_str: str) -> int:
    SUFFIX_TO_FACTOR = {
        "m": 1e6,
        "b": 1e9,
        "t": 1e12,
    }
    n_params_str = n_params_str.lower()
    match = re.match(n_params_pattern, n_params_str)

    if not match:
        raise ValueError(f"unable to parse n_params str {n_params_str}")

    prefix = float(match.group(1))

    suffix = match.group(2).lower()
    if suffix not in SUFFIX_TO_FACTOR:
        raise ValueError(
            f"unable to parse n_params str '{n_params_str}' because of unrecognized suffix '{suffix}'"
        )
    factor = SUFFIX_TO_FACTOR[suffix]

    return int(prefix * factor)


def parse_n_accelerators(n_accelerators_str: str) -> int:
    n_accelerators_str = n_accelerators_str.lower()
    if n_accelerators_str.isnumeric():
        return int(n_accelerators_str)
    elif "x" in n_accelerators_str:
        # axb format
        match = re.match(n_accelerators_pattern, n_accelerators_str)
        multiplicand_1 = int(match.group(1))
        multiplicand_2 = int(match.group(2))

        return multiplicand_1 * multiplicand_2
    else:
        raise ValueError(f"unable to parse n_accelerators input '{n_accelerators_str}'")


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "-s",
        "--samples-per-sec",
        type=float,
        required=True,
        help="training throughput in samples/sec",
    )
    parser.add_argument(
        "-l",
        "--seqlen",
        type=int,
        required=True,
        help="sequence length in tokens (i.e. tokens/sample)",
    )
    parser.add_argument(
        "-m",
        "--model-size",
        type=str,
        required=True,
        help="model size (for example 100m, 100b, or 100t)",
    )
    parser.add_argument(
        "-n",
        "--n-accelerators",
        type=str,
        required=True,
        help="number of accelerators used for training. accepts formats like '16' or '2x8'",
    )
    parser.add_argument(
        "-t",
        "--tflops-per-accelerator",
        type=float,
        required=True,
        help="theoretical tflops per accelerator",
    )
    args = parser.parse_args()

    samples_per_sec = args.samples_per_sec
    tokens_per_sample = args.seqlen
    model_size = parse_n_params(args.model_size)
    n_accelerators = parse_n_accelerators(args.n_accelerators)
    flops_per_accelerator = args.tflops_per_accelerator * 1e12

    tokens_per_sec = tokens_per_sample * samples_per_sec
    flops_per_token = 6 * model_size

    # MFU
    achieved_flops = flops_per_token * tokens_per_sec
    theoretical_flops = flops_per_accelerator * n_accelerators
    mfu = achieved_flops / theoretical_flops

    # tokens/day
    tokens_per_day = tokens_per_sec * 60 * 60 * 24

    print(
        f"inputs:\n"
        f"* model size: {model_size * 1e-9:.2f}B parameters\n"
        f"* sequence length: {tokens_per_sample}\n"
        f"* training throughput: {samples_per_sec:.2f} samples/second\n"
        f"* n_accelerators: {n_accelerators}\n"
        f"* FLOPs/accelerator: {flops_per_accelerator * 1e-12:.2f} TFLOPs\n"
    )

    print(f"{tokens_per_day * 1e-9:.2f}B tokens/day")
    print(f"{mfu * 100:.2f}% MFU")


if __name__ == "__main__":
    main()
