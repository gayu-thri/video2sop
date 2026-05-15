import json
from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL


@dataclass(frozen=True)
class DownloadResult:
    """
    Holds paths and metadata for a downloaded video.
    """

    video_path: Path
    audio_path: Path
    title: str
    duration_seconds: float
    source_url: str


def download(url: str, dest: Path) -> DownloadResult:
    """
    Downloads a YouTube video as mp4 and 16 kHz mono wav.

    Parameters
    ----------
    url: `str`
        YouTube video URL to download
    dest: `Path`
        Directory to save the downloaded files into

    Returns
    -------
    DownloadResult with file paths and video metadata
    """
    dest.mkdir(parents=True, exist_ok=True)
    video_tmpl = str(dest / "video.%(ext)s")
    audio_tmpl = str(dest / "audio.%(ext)s")
    info_path = dest / "info.json"

    video_path = dest / "video.mp4"
    audio_path = dest / "audio.wav"

    if not video_path.exists():
        with YoutubeDL(
            {
                "outtmpl": video_tmpl,
                "format": "bv*[ext=mp4][height<=720]+ba[ext=m4a]/b[ext=mp4]/best",
                "merge_output_format": "mp4",
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "concurrent_fragment_downloads": 4,
            }
        ) as ydl:
            info = ydl.extract_info(url, download=True)
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
        info_path.write_text(json.dumps(_slim_info(info), indent=2))

    if not audio_path.exists():
        with YoutubeDL(
            {
                "outtmpl": audio_tmpl,
                "format": "ba/b",
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "0",
                    }
                ],
                "postprocessor_args": ["-ar", "16000", "-ac", "1"],
            }
        ) as ydl:
            ydl.download([url])

    info_data = json.loads(info_path.read_text()) if info_path.exists() else {}
    return DownloadResult(
        video_path=video_path,
        audio_path=audio_path,
        title=info_data.get("title", "Untitled"),
        duration_seconds=float(info_data.get("duration") or 0.0),
        source_url=info_data.get("webpage_url", url),
    )


def _slim_info(info: dict) -> dict:
    """
    Keeps only the useful fields from a yt-dlp info dict.

    Parameters
    ----------
    info: `dict`
    Raw metadata dict returned by yt-dlp

    Returns
    -------
    Trimmed dict with title, duration, and URL
    """
    keep = ("id", "title", "duration", "webpage_url", "uploader", "upload_date")
    return {k: info.get(k) for k in keep if k in info}
