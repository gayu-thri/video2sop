import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from video2sop.config import Settings
from video2sop.document import render
from video2sop.downloader import download
from video2sop.frames import extract_keyframes
from video2sop.sop_generator import generate_instruction
from video2sop.transcriber import transcribe
from video2sop.vision import describe_frames

AudioMode = Literal["auto", "audio", "vision"]
log = logging.getLogger("video2sop")


@dataclass(frozen=True)
class Task:
    """
    One video processing job loaded from tasks.yaml.
    """

    id: str
    url: str
    asset_name: str
    activity_name: str
    audio_mode: AudioMode = "auto"


@dataclass(frozen=True)
class TaskResult:
    """
    Paths to the files produced for a completed task.
    """

    task_id: str
    markdown_path: Path


def run_task(task: Task, settings: Settings) -> TaskResult:
    """
    Runs all five pipeline stages for a single task.

    Parameters
    ----------
    task: `Task`
        The video job to process
    settings: `Settings`
        Runtime configuration and output paths

    Returns
    -------
    TaskResult with the path to the written Markdown file
    """
    log.info("=== %s — %s: %s ===", task.id, task.asset_name, task.activity_name)
    workdir = settings.workdir / task.id
    workdir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading video and audio...")
    dl = download(task.url, workdir / "media")
    log.info("  title=%r duration=%.0fs", dl.title, dl.duration_seconds)

    log.info("Extracting keyframes...")
    keyframes = extract_keyframes(
        dl.video_path,
        out_dir=workdir / "frames",
        max_frames=settings.max_keyframes,
        sample_fps=settings.sample_fps,
        clip_model=settings.clip_model,
        clip_pretrained=settings.clip_pretrained,
        clip_device=settings.clip_device,
    )
    log.info("  %d keyframes extracted", len(keyframes))

    transcript = None
    frame_descs = None

    if task.audio_mode in ("audio", "auto"):
        log.info("Transcribing audio with faster-whisper (%s)...", settings.whisper_model)
        transcript = transcribe(
            dl.audio_path,
            model=settings.whisper_model,
            compute_type=settings.whisper_compute,
            cache_dir=workdir / "transcript",
        )
        log.info(
            "  language=%s segments=%d meaningful=%s",
            transcript.language, len(transcript.segments), transcript.is_meaningful,
        )

    use_vision = task.audio_mode == "vision" or (
        task.audio_mode == "auto" and (transcript is None or not transcript.is_meaningful)
    )
    if use_vision:
        log.info("Describing keyframes with llava...")
        frame_descs = describe_frames(
            keyframes,
            asset_name=task.asset_name,
            activity_name=task.activity_name,
            model=settings.vision_model,
            host=settings.ollama_host,
            cache_path=workdir / "frame_descriptions.json",
        )
        log.info("  %d descriptions generated", len(frame_descs))

    log.info("Generating work instruction with %s...", settings.ollama_model)
    doc = generate_instruction(
        asset_name=task.asset_name,
        activity_name=task.activity_name,
        source_url=dl.source_url,
        transcript=transcript if not use_vision else None,
        frame_descriptions=frame_descs,
        keyframes=keyframes,
        model=settings.ollama_model,
        host=settings.ollama_host,
        cache_path=workdir / "instruction.json",
    )
    log.info("  %d steps", len(doc.steps))

    if use_vision and keyframes:
        doc = _fill_missing_frames(doc, keyframes)

    log.info("Rendering Markdown...")
    basename = _safe_basename(task.asset_name, task.activity_name)
    md_path = render(
        doc,
        keyframes=keyframes,
        source_url=dl.source_url,
        out_dir=settings.outdir,
        doc_basename=basename,
    )
    log.info("  %s", md_path)
    return TaskResult(task_id=task.id, markdown_path=md_path)


def _fill_missing_frames(doc, keyframes) -> "WorkInstruction":
    """
    Assigns a keyframe to every step that got frame_index=-1.

    Parameters
    ----------
    doc: `WorkInstruction`
        Instruction with potentially unassigned frame indices
    keyframes: `list[Keyframe]`
        All available keyframes to assign from

    Returns
    -------
    New WorkInstruction with a frame assigned to every step
    """
    from video2sop.sop_generator import Step, WorkInstruction

    valid_indices = {kf.index for kf in keyframes}
    n_steps = len(doc.steps)
    n_frames = len(keyframes)
    fixed_steps = []
    for i, step in enumerate(doc.steps):
        if step.frame_index in valid_indices:
            fixed_steps.append(step)
        else:
            frame_pos = round(i / max(n_steps - 1, 1) * (n_frames - 1))
            fallback_index = keyframes[frame_pos].index
            fixed_steps.append(Step(
                number=step.number,
                title=step.title,
                description=step.description,
                frame_index=fallback_index,
            ))
    return WorkInstruction(title=doc.title, overview=doc.overview, steps=fixed_steps)


def _safe_basename(asset: str, activity: str) -> str:
    """
    Turns asset and activity names into a safe filename stem.

    Parameters
    ---------------
    asset: `str`
        Equipment name
    activity: `str`
        Activity name

    Returns
    -------
    Lowercase alphanumeric string safe for use as a filename
    """
    raw = f"{asset}__{activity}".lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
