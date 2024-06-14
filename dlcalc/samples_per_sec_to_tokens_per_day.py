"""simple utility to convert samples/second to tokens/day"""

from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "-t",
        "--samples-per-sec",
        type=float,
        required=True,
        help="training throughput in samples/sec",
    )
    parser.add_argument(
        "-s",
        "--seqlen",
        type=int,
        required=True,
        help="sequence length in tokens (i.e. tokens/sample)",
    )
    args = parser.parse_args()

    samples_per_sec = args.samples_per_sec
    tokens_per_sample = args.seqlen

    tokens_per_day = samples_per_sec * tokens_per_sample * 60 * 60 * 24

    print(
        f"{samples_per_sec} samples/sec "
        f"with sequence length {tokens_per_sample} "
        f"translates to {tokens_per_day * 1e-9:.2f}B tokens/day"
    )


if __name__ == "__main__":
    main()
