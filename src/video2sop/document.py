import shutil
from pathlib import Path

from video2sop.frames import Keyframe
from video2sop.sop_generator import WorkInstruction


def render(
    doc: WorkInstruction,
    *,
    keyframes: list[Keyframe],
    source_url: str,
    out_dir: Path,
    doc_basename: str,
) -> Path:
    """
    Writes the work instruction to a Markdown file with inline screenshots.

    Parameters
    ----------
    doc: `WorkInstruction`
        The structured instruction to render
    keyframes: `list[Keyframe]`
        All keyframes so images can be looked up by index
    source_url: `str`
        Video URL included as attribution in the document
    out_dir: `Path`
        Directory to write the Markdown and images folder into
    doc_basename: `str`
        Filename stem for the .md file and images folder

    Returns
    -------
    Path to the written Markdown file
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / f"{doc_basename}_images"
    images_dir.mkdir(exist_ok=True)
    frames_by_index = {kf.index: kf for kf in keyframes}

    md_path = out_dir / f"{doc_basename}.md"
    lines: list[str] = [f"# {doc.title}\n", f"**Source:** {source_url}\n"]

    if doc.overview:
        lines.append("## Overview\n")
        lines.append(doc.overview + "\n")

    lines.append("## Steps\n")
    for step in doc.steps:
        lines.append(f"### Step {step.number}. {step.title}\n")
        lines.append(step.description + "\n")
        kf = frames_by_index.get(step.frame_index)
        if kf and kf.image_path.exists():
            dest = images_dir / kf.image_path.name
            if not dest.exists():
                shutil.copy2(kf.image_path, dest)
            rel = dest.relative_to(md_path.parent)
            lines.append(f"![Step {step.number}]({rel})\n")
            lines.append(f"*Frame at {kf.timestamp:.1f}s.*\n")

    md_path.write_text("\n".join(lines))
    return md_path
