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
    parser.add_argument(
        "-m",
        "--model-size-in-b",
        type=float,
        required=True,
        help="model size (in billions of parameters)",
    )
    parser.add_argument(
        "-n",
        "--n-nodes",
        type=int,
        required=True,
        help="number of nodes used for training",
    )
    parser.add_argument(
        "--tflops-per-node",
        type=float,
        required=True,
        help="theoretical tflops per node",
    )
    args = parser.parse_args()

    samples_per_sec = args.samples_per_sec
    tokens_per_sample = args.seqlen
    model_size = args.model_size_in_b * 1e9
    n_nodes = args.n_nodes
    flops_per_node = args.tflops_per_node * 1e12

    achieved_flops = samples_per_sec * tokens_per_sample * 6 * model_size
    theoretical_flops = flops_per_node * n_nodes

    mfu = achieved_flops / theoretical_flops

    print(
        f"{samples_per_sec} samples/sec "
        f"with sequence length {tokens_per_sample} "
        f"and model size {model_size * 1e-9:.2f}B "
        f"translates to {mfu * 100:.2f}% MFU"
    )


if __name__ == "__main__":
    main()
