import json
from argparse import ArgumentParser

from . import traces


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "trace_path",
        type=str,
        help="Path to the trace file to process.",
    )
    parser.add_argument(
        "--no-python",
        action="store_true",
        help="Filter Python stacktraces from the trace file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output trace file.",
    )
    args = parser.parse_args()
    trace_path = args.trace_path
    out_path = args.output
    no_python = args.no_python

    if not out_path:
        out_path = trace_path

    with open(trace_path) as f:
        trace = json.load(f)

    trace = traces.filter_events(trace)
    if no_python:
        trace = traces.drop_python_stacktraces(trace)
    trace = traces.move_to_reasonable_streams(trace)

    with open(out_path, "w") as f:
        json.dump(trace, f)


if __name__ == "__main__":
    main()
