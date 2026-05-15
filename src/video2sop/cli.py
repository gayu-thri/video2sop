import argparse
import logging
import sys
from pathlib import Path

import yaml

from video2sop.config import load_settings
from video2sop.pipeline import Task, run_task


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the video2sop command.

    Parameters
    ----------
    argv: `list[str] | None`
        Command-line arguments; uses sys.argv if None

    Returns
    -------
    Exit code: 0 on success, non-zero on failure
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        prog="video2sop",
        description="Generate work instructions from instructional videos.",
    )
    parser.add_argument(
        "-t", "--tasks", type=Path, default=Path("tasks.yaml"),
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    tasks = _load_tasks(args.tasks)

    try:
        for task in tasks:
            run_task(task, settings)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        return 1

    return 0


def _load_tasks(path: Path) -> list[Task]:
    """
    Parses tasks.yaml into a list of Task objects.

    Parameters
    ----------
    path: `Path`
        Path to the YAML tasks file to read

    Returns
    -------
    List of Task objects ready to run
    """
    if not path.exists():
        raise SystemExit(f"Tasks file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    return [
        Task(
            id=t["id"],
            url=t["url"],
            asset_name=t["asset_name"],
            activity_name=t["activity_name"],
            audio_mode=t.get("audio_mode", "auto"),
        )
        for t in data.get("tasks", [])
    ]


if __name__ == "__main__":
    sys.exit(main())
