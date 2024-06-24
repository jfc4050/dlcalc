"""simple utility to convert samples/second to MFU."""

from argparse import ArgumentParser


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
        "--model-size-in-b",
        type=float,
        required=True,
        help="model size (in billions of parameters)",
    )
    parser.add_argument(
        "-n",
        "--n-accelerators",
        type=int,
        required=True,
        help="number of accelerators used for training",
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
    model_size = args.model_size_in_b * 1e9
    n_accelerators = args.n_accelerators
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
