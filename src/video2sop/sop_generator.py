import json
from pathlib import Path

import ollama
from pydantic import BaseModel

from video2sop.frames import Keyframe
from video2sop.transcriber import Transcript
from video2sop.vision import FrameDescription


class Step(BaseModel):
    """
    One numbered step in the generated work instruction.
    """

    number: int
    title: str
    description: str
    frame_index: int = -1


class WorkInstruction(BaseModel):
    """
    The complete structured work instruction with title, overview, and steps.
    """

    title: str
    overview: str
    steps: list[Step]


def generate_instruction(
    *,
    asset_name: str,
    activity_name: str,
    source_url: str,
    transcript: Transcript | None,
    frame_descriptions: list[FrameDescription] | None,
    keyframes: list[Keyframe],
    model: str,
    host: str,
    cache_path: Path,
) -> WorkInstruction:
    """
    Calls llama3.2 to produce a structured work instruction from source material.

    Parameters
    ----------
    asset_name: `str`
        Name of the equipment being worked on
    activity_name: `str`
        Name of the activity being performed
    source_url: `str`
        Original video URL included for attribution
    transcript: `Transcript | None`
        Whisper transcript when audio is available
    frame_descriptions: `list[FrameDescription] | None`
        llava descriptions for vision-mode tasks
    keyframes: `list[Keyframe]`
        All selected keyframes with their timestamps
    model: `str`
        Ollama text model to call
    host: `str`
        Ollama server base URL
    cache_path: `Path`
        Where to read or write the cached instruction

    Returns
    -------
    Validated WorkInstruction with title, overview, and steps
    """
    if cache_path.exists():
        return WorkInstruction.model_validate_json(cache_path.read_text())

    prompt = _build_prompt(
        asset_name=asset_name,
        activity_name=activity_name,
        source_url=source_url,
        transcript=transcript,
        frame_descriptions=frame_descriptions,
        keyframes=keyframes,
    )

    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=WorkInstruction.model_json_schema(),
    )

    doc = WorkInstruction.model_validate_json(response.message.content)
    cache_path.write_text(doc.model_dump_json(indent=2))
    return doc


def _build_prompt(
    *,
    asset_name: str,
    activity_name: str,
    source_url: str,
    transcript: Transcript | None,
    frame_descriptions: list[FrameDescription] | None,
    keyframes: list[Keyframe],
) -> str:
    """
    Assembles the text prompt sent to llama3.2.

    Parameters
    ----------
    asset_name:`str`
        Equipment name for the instruction header
    activity_name:`str`
        Activity name for the instruction header
    source_url:`str`
        Video URL included for reference
    transcript:`Transcript | None`
        Spoken content to include when available
    frame_descriptions:`list[FrameDescription] | None`
        Visual descriptions for vision-mode tasks
    keyframes:`list[Keyframe]`
        Keyframe list with timestamps for alignment

    Returns
    -------
    Formatted prompt string ready to send to the model
    """
    parts: list[str] = [
        f"You are a senior technical writer producing a maintenance manual.",
        f"Your task: write a professional work instruction for '{activity_name}' on a '{asset_name}'.",
        f"The output must read like a real maintenance SOP, not a description of a video.",
        "",
        "CRITICAL RULES — violations will make the output unusable:",
        "1. Every title and description must be direct imperative voice addressed to the person performing the work.",
        "   WRONG: 'The technician inspects the valve for wear.'",
        "   RIGHT:  'Inspect the valve seat and stem for scoring, corrosion, or uneven wear.'",
        "   Never write 'the technician', 'the operator', 'the image shows', or any third-person phrasing.",
        "2. Do not reference frames, images, screenshots, or the video in step text.",
        "3. Each step: one discrete, verifiable action.",
        "4. Step description must be 2-4 sentences covering:",
        "   a) the specific action and component (use the component name from the source material),",
        "   b) what to check, measure, or observe,",
        "   c) the acceptance criterion or expected condition (e.g. 'no cracks', 'indicator in green range'),",
        "   d) what to do if the condition is not met, where determinable from the source material.",
        "5. Only include information supported by the source material.",
        "6. For each step, set frame_index to the keyframe index that best illustrates it (use -1 if nothing fits).",
        "7. Order steps in the sequence a technician would follow.",
        "8. Aim for 8-15 steps.",
        "",
        f"Equipment: {asset_name}",
        f"Activity: {activity_name}",
        "",
    ]

    if transcript and transcript.is_meaningful:
        parts.append("=== AUDIO TRANSCRIPT (timestamps in seconds) ===")
        for seg in transcript.segments:
            parts.append(f"[{seg.start:7.2f} - {seg.end:7.2f}] {seg.text}")
        parts.append("")

    if frame_descriptions:
        parts.append("=== KEYFRAME DESCRIPTIONS (visual, in temporal order) ===")
        for fd in frame_descriptions:
            parts.append(f"Frame {fd.index} @ t={fd.timestamp:.1f}s: {fd.description}")
    else:
        parts.append("=== AVAILABLE KEYFRAMES ===")
        parts.append(
            "To assign frame_index for each step: identify the transcript timestamp range "
            "that corresponds to the step, then pick the keyframe whose timestamp falls "
            "closest to the midpoint of that range."
        )
        for kf in keyframes:
            parts.append(f"Frame {kf.index} @ t={kf.timestamp:.1f}s")
    parts.append("")
    parts.append("Output the JSON now.")

    return "\n".join(parts)
