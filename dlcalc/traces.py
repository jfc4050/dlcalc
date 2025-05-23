COMPUTE_TID = 1
TP_COMM_TID = 2
SEND_RECV_TID = 3
DP_COMM_TID = 4
MEM_TID = 5


def filter_events(trace: dict) -> dict:
    updated_events = []
    for event in trace["traceEvents"]:
        event: dict
        # takes up space in kernel view
        if event.get("cat") == "python_function" and event["tid"] == 0:
            continue
        # get rid of ProfilerStep annotations
        if event.get("cat") in ("user_annotation", "gpu_user_annotation") and event[
            "name"
        ].startswith("ProfilerStep"):
            continue
        if event.get("cat") == "Trace" and event["name"].startswith("PyTorch Profiler"):
            continue

        updated_events.append(event)

    trace["traceEvents"] = updated_events

    return trace


def move_to_reasonable_streams(trace: dict) -> dict:
    # flow events need to be tracked separately
    correlation_id_to_new_tid = {}

    # pass 1
    updated_events = []
    for event in trace["traceEvents"]:
        event: dict

        # move stuff into more reasonable streams
        if event.get("cat") == "kernel":
            new_tid = COMPUTE_TID
            event_name: str = event["name"]
            correlation_id: int = event["args"]["correlation"]
            if event_name.startswith(("ncclDevKernel_AllGather", "ncclDevKernel_ReduceScatter")):
                new_tid = TP_COMM_TID
            elif event_name.startswith("ncclDevKernel_SendRecv"):
                new_tid = SEND_RECV_TID

            correlation_id_to_new_tid[correlation_id] = new_tid
            event["tid"] = new_tid

        elif event.get("cat") == "gpu_user_annotation":
            new_tid = COMPUTE_TID
            event_name: str = event["name"]
            if event_name.startswith(
                (
                    "nccl:all_gather",
                    "nccl:reduce_scatter",
                    "nccl:_all_gather",
                    "nccl:_reduce_scatter",
                )
            ):
                new_tid = TP_COMM_TID
            elif event_name.startswith(("nccl:send", "nccl:recv")):
                new_tid = SEND_RECV_TID
            event["tid"] = new_tid

        elif event.get("cat") in ("gpu_memcpy", "gpu_memset"):
            correlation_id: int = event["args"]["correlation"]
            correlation_id_to_new_tid[correlation_id] = MEM_TID
            event["tid"] = MEM_TID

        updated_events.append(event)

    trace["traceEvents"] = updated_events

    # pass 2
    updated_events = []
    for event in trace["traceEvents"]:
        if event.get("cat") == "ac2g":
            correlation_id: int = event["id"]
            if event.get("bp") == "e":
                if correlation_id in correlation_id_to_new_tid:
                    event["tid"] = correlation_id_to_new_tid[correlation_id]
        updated_events.append(event)

    trace["traceEvents"] = updated_events

    return trace
