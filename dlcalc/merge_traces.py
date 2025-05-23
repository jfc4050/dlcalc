import json
from argparse import ArgumentParser

from . import traces


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "trace_paths",
        nargs="+",
        type=str,
        help="Path to the trace file to process.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output trace file.",
    )
    args = parser.parse_args()
    trace_paths = args.trace_paths
    out_path = args.output

    merged_trace: dict = None  # type: ignore
    for idx, trace_path in enumerate(trace_paths):
        with open(trace_path) as f:
            trace = json.load(f)

        trace = traces.filter_events(trace)
        trace = traces.drop_python_stacktraces(trace)
        trace = traces.update_pid_with_rank(trace, new_rank=idx)
        trace = traces.move_to_reasonable_streams(trace)

        if merged_trace is None:
            merged_trace = trace
        else:
            merged_trace["traceEvents"].extend(trace["traceEvents"])

    with open(out_path, "w") as f:
        json.dump(merged_trace, f)


if __name__ == "__main__":
    main()
