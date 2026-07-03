import json
import warnings
from typing import List, Tuple


def load_records(path: str) -> List[dict]:
    """
    Load:
      - JSONL: each line is a JSON object
      - JSON array: [ {...}, {...} ]
      - single JSON object: {...}
    Return: list of records
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []

    # JSON array or object
    if text[0] in ["[", "{"]:
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError as e:
            # Multi-line JSONL that starts with "{" often raises "Extra data" here.
            # Treat it as expected and continue silently to JSONL parsing.
            likely_jsonl = text[0] == "{" and e.msg == "Extra data"
            if not likely_jsonl:
                warnings.warn(
                    "[load_records] Failed to parse whole file as JSON "
                    f"for '{path}' ({e.msg} at line {e.lineno}, col {e.colno}); "
                    "falling back to JSONL parsing.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # JSONL fallback
    out = []
    for ln, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[load_records] Invalid JSONL in '{path}' at line {ln}: "
                f"{e.msg} (col {e.colno})"
            ) from e
    return out


def record_to_jobs(record: dict) -> Tuple[float, List[dict]]:
    dispatch_time = float(record.get("dispatch_time", 0.0))
    jobs = record.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("record['jobs'] must be a list")
    return dispatch_time, jobs


def poll_live_job_file(fh) -> List[dict]:
    records: List[dict] = []
    while True:
        pos = fh.tell()
        line = fh.readline()
        if not line:
            fh.seek(pos)
            break
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(rec)
    return records
