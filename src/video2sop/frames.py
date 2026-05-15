import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Keyframe:
    """
    A selected keyframe with its timestamp and saved image path.
    """

    index: int
    timestamp: float
    image_path: Path

    def to_json(self) -> dict:
        """
        Serialises the keyframe to a plain dict for JSON storage.

        Returns
        -------
        Dict with index, timestamp, and image path as string
        """
        d = asdict(self)
        d["image_path"] = str(self.image_path)
        return d


def extract_keyframes(
    video_path: Path,
    *,
    out_dir: Path,
    max_frames: int,
    sample_fps: float,
    clip_model: str,
    clip_pretrained: str,
    clip_device: str | None = None,
    target_width: int = 960,
) -> list[Keyframe]:
    """
    Samples a video and picks the most visually diverse frames.

    Parameters
    ----------
    video_path: `Path`
            Path to the downloaded mp4 file
    out_dir - `Path`
            Directory to write frames and caches into
    max_frames: `int`
            Maximum number of keyframes to select
    sample_fps: `float`
            Candidate frames to consider per second
    clip_model: `str`
            OpenCLIP model architecture name
    clip_pretrained - `str`
            Pretrained weights variant to load
    clip_device: `str | None`
            Device to run CLIP on; auto-detected if None
    target_width: `int`
            Width in pixels to resize saved frames to

    Returns
    -------
    List of selected Keyframe objects in timestamp order
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keyframes_cache = out_dir / "keyframes.json"
    if keyframes_cache.exists():
        return _load_keyframes(keyframes_cache)

    embeddings_cache = out_dir / "embeddings.npz"
    if embeddings_cache.exists():
        with np.load(embeddings_cache) as data:
            timestamps = data["timestamps"].tolist()
            embeddings = data["embeddings"]
        log.info("  reusing %d cached candidate embeddings", len(timestamps))
    else:
        timestamps, embeddings = _embed_candidates(
            video_path,
            sample_fps=sample_fps,
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=clip_device,
        )
        np.savez(embeddings_cache, timestamps=np.array(timestamps), embeddings=embeddings)
        log.info("  embedded %d candidates", len(timestamps))

    if len(timestamps) == 0:
        keyframes_cache.write_text("[]")
        return []

    selected = _farthest_point_sample(embeddings, k=min(max_frames, len(embeddings)))
    selected_sorted = sorted(selected, key=lambda i: timestamps[i])
    keyframes = _save_selected_frames(
        video_path,
        timestamps=[timestamps[i] for i in selected_sorted],
        out_dir=out_dir,
        target_width=target_width,
    )

    keyframes_cache.write_text(json.dumps([k.to_json() for k in keyframes], indent=2))
    return keyframes


def _embed_candidates(
    video_path: Path,
    *,
    sample_fps: float,
    model_name: str,
    pretrained: str,
    device: str | None,
) -> tuple[list[float], np.ndarray]:
    """
    Samples frames from the video and returns their CLIP embeddings.

    Parameters
    ----------
    video_path: `Path`
            Path to the source video file
    sample_fps: `float`
            Candidate frames to extract per second
    model_name: `str`
            OpenCLIP architecture name
    pretrained: `str`
            Pretrained weights variant to use
    device: `str | None`
            Compute device; auto-selected if None

    Returns
    -------
    Tuple of timestamps list and L2-normalised embedding array
    """
    import cv2
    import open_clip
    import torch
    from PIL import Image

    chosen_device = device or _auto_device(torch)
    log.info("  loading CLIP %s/%s on %s", model_name, pretrained, chosen_device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=chosen_device
    )
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if fps > 0 else 0.0
        if duration <= 0:
            return [], np.zeros((0, 1), dtype=np.float32)

        n_samples = max(1, int(duration * sample_fps))
        sample_times = np.linspace(0.0, duration, n_samples, endpoint=False)

        all_embeds: list[np.ndarray] = []
        kept_times: list[float] = []
        batch_imgs: list[Image.Image] = []
        batch_times: list[float] = []
        BATCH = 32

        def flush() -> None:
            if not batch_imgs:
                return
            with torch.no_grad():
                t = torch.stack([preprocess(im) for im in batch_imgs]).to(chosen_device)
                e = model.encode_image(t)
                norms = e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                e = e / norms
                all_embeds.append(e.cpu().float().numpy())
            kept_times.extend(batch_times)
            batch_imgs.clear()
            batch_times.clear()

        for ts in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_imgs.append(Image.fromarray(rgb))
            batch_times.append(float(ts))
            if len(batch_imgs) >= BATCH:
                flush()
        flush()
    finally:
        cap.release()

    if not all_embeds:
        return [], np.zeros((0, 1), dtype=np.float32)
    return kept_times, np.concatenate(all_embeds, axis=0)


def _farthest_point_sample(embeds: np.ndarray, *, k: int) -> list[int]:
    """
    Picks k diverse indices from an L2-normalised embedding matrix.

    Parameters
    ----------
    embeds: `np.ndarray`
        L2-normalised embedding matrix of shape (n, d)
    k: `int`
        Number of points to select

    Returns
    -------
    List of selected row indices
    """
    n = len(embeds)
    if k >= n:
        return list(range(n))

    embeds = np.nan_to_num(embeds, nan=0.0, posinf=0.0, neginf=0.0)
    row_norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-8)
    embeds = embeds / row_norms

    centroid = embeds.mean(axis=0)
    norm = np.linalg.norm(centroid) + 1e-8
    centroid = centroid / norm

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        seed = int(np.argmax(embeds @ centroid))

        selected = [seed]
        min_dist = 1.0 - embeds @ embeds[seed]
        min_dist[seed] = -np.inf

        while len(selected) < k:
            nxt = int(np.argmax(min_dist))
            if not np.isfinite(min_dist[nxt]) or min_dist[nxt] <= 0:
                break
            selected.append(nxt)
            new_dist = 1.0 - embeds @ embeds[nxt]
            min_dist = np.minimum(min_dist, new_dist)
            min_dist[nxt] = -np.inf

    return selected


def _save_selected_frames(
    video_path: Path,
    *,
    timestamps: list[float],
    out_dir: Path,
    target_width: int,
) -> list[Keyframe]:
    """
    Seeks to each timestamp and writes a JPEG to disk.

    Parameters
    ----------
    video_path: `Path`
        Path to the source video file
    timestamps: `list[float]`
        Seconds at which to extract frames
    out_dir: `Path`
        Directory to write the JPEG files into
    target_width: `int`
        Maximum width to resize each frame to

    Returns
    -------
    List of Keyframe objects for the saved images
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    keyframes: list[Keyframe] = []
    try:
        for idx, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = _resize(frame, target_width, cv2)
            img_path = out_dir / f"frame_{idx:03d}.jpg"
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            keyframes.append(Keyframe(index=idx, timestamp=float(ts), image_path=img_path))
    finally:
        cap.release()
    return keyframes


def _resize(frame, target_width: int, cv2):
    """
    Scales a frame down to target_width, keeping aspect ratio.

    Parameters
    ----------
    frame: `np.ndarray`
        OpenCV BGR frame to resize
    target_width: `int`
        Maximum output width in pixels
    cv2: module
        OpenCV module reference

    Returns
    -------
    Resized frame, or original if already narrow enough
    """
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _auto_device(torch) -> str:
    """
    Picks the best available device for CLIP inference.

    Parameters
    ----------
    torch:
        PyTorch module reference

    Returns
    -------
    Device string: 'mps', 'cuda', or 'cpu'
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_keyframes(path: Path) -> list[Keyframe]:
    """
    Reads a saved keyframes.json back into Keyframe objects.

    Parameters
    ----------
    path: `Path`
        Path to the keyframes JSON cache file

    Returns
    -------
    List of Keyframe objects reconstructed from disk
    """
    data = json.loads(path.read_text())
    return [
        Keyframe(index=d["index"], timestamp=d["timestamp"], image_path=Path(d["image_path"]))
        for d in data
    ]
