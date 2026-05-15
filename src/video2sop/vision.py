import json
import logging
from dataclasses import dataclass
from pathlib import Path

import ollama

from video2sop.frames import Keyframe

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameDescription:
    """
    Holds the llava-generated description for one keyframe.
    """

    index: int
    timestamp: float
    image_path: Path
    description: str


def describe_frames(
    keyframes: list[Keyframe],
    *,
    asset_name: str,
    activity_name: str,
    model: str,
    host: str,
    cache_path: Path,
) -> list[FrameDescription]:
    """
    Asks llava to describe each keyframe individually via Ollama.

    Parameters
    ----------
    keyframes - `list[Keyframe]`
            Keyframes to describe in temporal order
    asset_name - `str`
            Equipment name used for context in the prompt
    activity_name - `str`
            Activity name used for context in the prompt
    model - `str`
            Ollama vision model to call
    host - `str`
            Ollama server base URL
    cache_path - `Path`
            Where to read or write the cached descriptions

    Returns
    -------
    List of FrameDescription objects in keyframe order
    """
    if cache_path.exists():
        return _load_cache(cache_path, keyframes)

    client = ollama.Client(host=host)
    out: list[FrameDescription] = []

    for i, kf in enumerate(keyframes, 1):
        log.info("  describing frame %d/%d (t=%.1fs)...", i, len(keyframes), kf.timestamp)
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"This is keyframe {kf.index} at t={kf.timestamp:.1f}s from a video "
                        f"showing '{activity_name}' on a '{asset_name}'. "
                        f"Write 1-2 sentences describing what the technician is doing "
                        f"or what state the equipment is in. Focus on observable actions, "
                        f"tools, and components. Do not invent anything not visible."
                    ),
                    "images": [str(kf.image_path)],
                }
            ],
        )
        out.append(
            FrameDescription(
                index=kf.index,
                timestamp=kf.timestamp,
                image_path=kf.image_path,
                description=response.message.content.strip(),
            )
        )

    cache_path.write_text(
        json.dumps(
            [{"index": d.index, "timestamp": d.timestamp, "description": d.description} for d in out],
            indent=2,
        )
    )
    return out


def _load_cache(path: Path, keyframes: list[Keyframe]) -> list[FrameDescription]:
    """
    Reads cached frame descriptions from disk.

    Parameters
    ----------
    path: `Path`
        Path to the cached JSON file
    keyframes: `list[Keyframe]`
        Keyframes to match descriptions against by index

    Returns
    -------
    List of FrameDescription objects loaded from disk
    """
    data = json.loads(path.read_text())
    by_index = {d["index"]: d["description"] for d in data}
    return [
        FrameDescription(
            index=kf.index,
            timestamp=kf.timestamp,
            image_path=kf.image_path,
            description=by_index.get(kf.index, ""),
        )
        for kf in keyframes
    ]
