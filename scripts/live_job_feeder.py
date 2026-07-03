import json
import random
import time
from pathlib import Path

# Append a new batch of jobs to live_jobs.jsonl every 2 seconds.
# Adjust JOBS_PER_BATCH, TYPES, and STATIONS as needed.

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "live_jobs.jsonl"
INTERVAL_SEC = 2.0
JOBS_PER_BATCH = 5
TYPES = ["A", "B", "C"]
STATIONS = [1, 2, 3, 4, 5]
DOCKS = [1, 2, 3, 4, 5]
# Phase III contract: material A/B/C -> duration 5/10/15
PROC_TIME_BY_TYPE = {"A": 5.0, "B": 10.0, "C": 15.0}

# If True, dispatch_time uses wall-clock elapsed seconds since start.
# If False, dispatch_time is always 0.0 (env will clamp to current time).
USE_ELAPSED_TIME = True


def make_job(jid: int) -> dict:
    jtype = random.choice(TYPES)
    return {
        "jid": jid,
        "type": jtype,
        "proc_time": float(PROC_TIME_BY_TYPE.get(jtype, 5.0)),
        "station": int(random.choice(STATIONS)),
        "dock": int(random.choice(DOCKS)),
    }


def append_batch(fh, batch_id: int, base_jid: int, dispatch_time: float) -> int:
    jobs = [make_job(base_jid + i) for i in range(JOBS_PER_BATCH)]
    rec = {
        "dispatch_time": float(dispatch_time),
        "jobs": jobs,
    }
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    return base_jid + JOBS_PER_BATCH


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    batch_id = 0
    next_jid = 0
    start = time.time()

    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        print(f"Appending to {OUTPUT_FILE} every {INTERVAL_SEC}s. Ctrl+C to stop.")
        try:
            while True:
                now = time.time()
                dispatch_time = (now - start) if USE_ELAPSED_TIME else 0.0
                next_jid = append_batch(fh, batch_id, next_jid, dispatch_time)
                batch_id += 1
                time.sleep(INTERVAL_SEC)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
