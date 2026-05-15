# video2sop

Takes a YouTube link to an instructional video and writes out a Markdown work
instruction with screenshots. One video in, one .md file out.

Built for the eqrx assessment. The two sample tasks are:

- Progressive Cavity Pump disassembly: Has spoken commentary, so the pipeline
  transcribes the audio and builds steps from what the technician says
- CAPS 1200 EL visual inspection: Silent video, so the pipeline describes each
  keyframe visually and builds steps from those descriptions instead

**Note**: The PCP video is from the https://www.youtube.com/@newwasteconcepts channel.
Full channel processing is not supported. The pipeline takes one direct video
URL at a time.

## Setup

You need ffmpeg and Ollama on the system. No API key needed.

```bash
$ brew install ffmpeg
$ curl -fsSL https://ollama.com/install.sh | sh
$ ollama pull llama3.2
$ ollama pull llava
$ poetry install
```

`llama3.2` is used for writing the work instruction. `llava` is used for
describing keyframes when the video has no usable audio (vision mode).
If the video has a clear spoken commentary, only `llama3.2` is needed.

## Running it

```bash
$ poetry run video2sop
```

Processes every task in `tasks.yaml`. Output Markdown files land in `final-outputs/`.

Each task must be a direct video URL (`watch?v=...`). Channel or playlist URLs
are not supported.

## Sample outputs

Pre-generated outputs are committed under `final-outputs/` so you can inspect the
results without running the pipeline yourself.

- **`progressive_cavity_pump_disassembly.md`**: Work instruction for the Progressive Cavity Pump
  disassembly. Built from the Whisper transcript, so steps are grounded in what the technician
  says verbatim: tool sizes, part names, and actions.
- **`caps_1200_el_visual_check_and_periodic_inspection.md`**: Work instruction for the CAPS 1200 EL
  visual inspection. Built from llava keyframe descriptions (silent video), with screenshots
  embedded inline for each step.

## How it works

There are five stages. Each one writes its result to `intermediate-outputs/<task_id>/` and
short-circuits on re-run, so iterating is cheap.

```
1. yt-dlp        downloads video.mp4 and audio.wav (16 kHz mono)
2. CLIP          embeds frames sampled at 1 fps, picks K diverse keyframes
3.a. whisper       transcribes the audio (skipped if the video is silent)
3.b. llava         describes the keyframes (only when there is no transcript)
4. llama3.2      emits a structured work instruction as JSON
5. renderer      writes out the Markdown with images inline (vision mode only)
```

### Keyframe selection

Sample one frame per second from the video. Run each through OpenCLIP
(ViT-B/32, OpenAI weights). That gives every frame a 512-d embedding.
Pick the K most visually diverse frames using farthest-point sampling in
embedding space: start from the frame closest to the centroid, then
repeatedly add the frame whose nearest already-picked neighbour is farthest
away. Stop at K. Sort the result by timestamp and save those frames.

This avoids the usual scene-cut threshold tuning. It works on single-take
videos with slow pans, on multi-cut tutorials, and on talking-head shots
all the same way.

The CLIP encode runs on Apple MPS (the Mac GPU) automatically. About 5
seconds for a 10-minute video. CPU fallback works too.

### Why these specific tools

| What | Choice | Reason |
|---|---|---|
| Download | yt-dlp | Maintained fork of youtube-dl, handles current YouTube formats |
| Speech-to-text | faster-whisper, small.en, int8 on CPU | Local, fast on Mac, no CUDA needed |
| Keyframe selection | OpenCLIP ViT-B/32 + farthest-point sampling | No thresholds to tune, works on any video, runs on Mac GPU |
| Frame description | llava via Ollama, one call per frame | Reliable with local vision models, results cached |
| Instruction synthesis | llama3.2 via Ollama, structured JSON mode | Forces typed JSON output, no fragile text parsing |
| Output | Plain Markdown | Portable, easy to review, opens in any editor |

## Project layout

```
pyproject.toml          poetry config
tasks.yaml              videos to process
src/video2sop/
  cli.py                entry point
  pipeline.py           runs the five stages in order
  config.py             settings
  downloader.py         yt-dlp wrapper
  transcriber.py        faster-whisper wrapper
  frames.py             CLIP embedding + farthest-point selection
  vision.py             llava vision, one call per frame
  sop_generator.py      llama3.2 JSON mode, returns Pydantic-validated JSON
  document.py           Markdown renderer
intermediate-outputs/   cached intermediates
final-outputs/          final Markdown files
```

## Known limits

Ollama local models (llava for vision, llama3.2 for text) were used so there
are no API limits or compute costs while running. The tradeoff is the output
quality is slightly weaker, particularly on the silent CAPS video. There is
clear scope to improve this by swapping in stronger models like Claude 3.5
Sonnet, GPT-4o, or Gemini 1.5 Pro for both vision and instruction generation.

The PCP disassembly output is good because the video has clear spoken
commentary. Whisper picks up tool sizes, part names, and actions verbatim
from the audio, so the steps are specific and grounded.

The CAPS 1200 EL output is weaker. That video is silent, so the pipeline
falls back to llava describing each keyframe. llava is a general-purpose 7B
vision model and struggles with specialist industrial equipment, it tends
to produce vague, repetitive descriptions that don't reflect what is actually
shown. The resulting work instruction inherits those problems.

