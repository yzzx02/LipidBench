from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(args: argparse.Namespace) -> None:
    queue_state = args.queue_state.resolve()
    lock_path = queue_state.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"queue lock already exists: {lock_path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("utf-8"))
        os.close(descriptor)
        write_json(
            queue_state,
            {
                "status": "waiting_for_seed_lodo",
                "pid": os.getpid(),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "seed_lodo_state": str(args.seed_lodo_state.resolve()),
                "detection_pipeline": str(args.detection_pipeline.resolve()),
            },
        )
        while True:
            if args.seed_lodo_state.exists():
                seed_state = read_json(args.seed_lodo_state)
                if seed_state.get("status") == "complete":
                    break
            time.sleep(30)

        write_json(
            queue_state,
            {
                "status": "running_detection_pipeline",
                "pid": os.getpid(),
                "detection_started_utc": datetime.now(timezone.utc).isoformat(),
                "seed_lodo_status": "complete",
            },
        )
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(args.detection_pipeline.resolve())]
        with args.log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write("\nCOMMAND: " + subprocess.list2cmdline(command) + "\n")
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return_code = process.wait()
        if return_code != 0:
            write_json(
                queue_state,
                {
                    "status": "detection_pipeline_failed",
                    "return_code": return_code,
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "log": str(args.log_path.resolve()),
                },
            )
            raise subprocess.CalledProcessError(return_code, command)
        write_json(
            queue_state,
            {
                "status": "complete",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "log": str(args.log_path.resolve()),
            },
        )
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-lodo-state",
        type=Path,
        default=Path(r"D:\CODE\LipidBench\PeakTruthLab\results\rtx4070_final_merged_20260814\lodo_concat_seed_20260814\PIPELINE_STATE.json"),
    )
    parser.add_argument(
        "--detection-pipeline",
        type=Path,
        default=Path(r"D:\CODE\LipidBench\PeakTruthLab\scripts\detection\run_rtx4070_multitask_concat_pipeline.py"),
    )
    parser.add_argument(
        "--queue-state",
        type=Path,
        default=Path(r"D:\CODE\LipidBench\PeakTruthLab\results\rtx4070_final_merged_20260814\multitask_concat_detection_seed_20260815\QUEUE_STATE.json"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(r"D:\CODE\LipidBench\PeakTruthLab\results\rtx4070_final_merged_20260814\multitask_concat_detection_seed_20260815\queue_console.log"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
