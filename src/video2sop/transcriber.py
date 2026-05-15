import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptSegment:
    """
    One timed chunk of spoken text from Whisper.
    """

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """
    Full transcription result with detected language and all segments.
    """

    language: str
    segments: list[TranscriptSegment]

    @property
    def full_text(self) -> str:
        """
        All segment text joined into one string.

        Returns
        -------
        Plain string of the complete spoken content
        """
        return " ".join(s.text.strip() for s in self.segments).strip()

    @property
    def is_meaningful(self) -> bool:
        """
        True when the transcript has enough words to be usable.

        Returns
        -------
        False for silent or near-silent videos
        """
        return len(self.full_text.split()) >= 12


def transcribe(audio_path: Path, *, model: str, compute_type: str, cache_dir: Path) -> Transcript:
    """
    Runs faster-whisper on an audio file and caches the result.

    Parameters
    ----------
    audio_path: `Path`
        Path to the 16 kHz mono wav file
    model: `str`
        Whisper model size to load
    compute_type: `str`
        Quantisation mode for the model
    cache_dir: `Path`
        Where to read or write the cached transcript

    Returns
    -------
    Transcript with detected language and timed segments
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "transcript.json"
    if cache_file.exists():
        return _load_cache(cache_file)

    from faster_whisper import WhisperModel

    whisper = WhisperModel(model, device="cpu", compute_type=compute_type)
    segments_iter, info = whisper.transcribe(
        str(audio_path),
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=1,
        condition_on_previous_text=False,
    )
    segments = [
        TranscriptSegment(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in segments_iter
        if (s.text or "").strip()
    ]
    transcript = Transcript(language=info.language, segments=segments)
    _write_cache(cache_file, transcript)
    return transcript


def _write_cache(path: Path, transcript: Transcript) -> None:
    """
    Serialises a transcript to JSON on disk.

    Parameters
    ----------
    path: `Path`
        File path to write to
    transcript: `Transcript`
        Transcript object to serialise
    """
    path.write_text(
        json.dumps(
            {
                "language": transcript.language,
                "segments": [asdict(s) for s in transcript.segments],
            },
            indent=2,
        )
    )


def _load_cache(path: Path) -> Transcript:
    """
    Reads a cached transcript back from JSON.

    Parameters
    ----------
    path: `Path`
        Path to the cached transcript file

    Returns
    -------
    Transcript reconstructed from disk
    """
    data = json.loads(path.read_text())
    return Transcript(
        language=data["language"],
        segments=[TranscriptSegment(**s) for s in data["segments"]],
    )
