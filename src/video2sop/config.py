import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

AudioMode = Literal["auto", "audio", "vision"]


@dataclass(frozen=True)
class Settings:
    """
    All runtime settings loaded once at startup.
    """

    ollama_host: str
    whisper_model: str
    whisper_compute: str
    ollama_model: str
    vision_model: str
    clip_model: str
    clip_pretrained: str
    clip_device: str | None
    sample_fps: float
    max_keyframes: int
    workdir: Path
    outdir: Path


def load_settings() -> Settings:
    """
    Builds a Settings object from environment variables and defaults.

    Returns
    -------
    Settings instance ready to pass into the pipeline
    """
    device = os.getenv("VIDEO2SOP_CLIP_DEVICE", "").strip() or None
    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        whisper_model="small.en",
        whisper_compute="int8",
        ollama_model="llama3.2",
        vision_model="llava",
        clip_model="ViT-B-32",
        clip_pretrained="openai",
        clip_device=device,
        sample_fps=1.0,
        max_keyframes=24,
        workdir=Path("work").resolve(),
        outdir=Path("output").resolve(),
    )
