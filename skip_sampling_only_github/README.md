# Semantic skip sampling

Frame selection for edge video: pretrained embeddings, YOLO class weights, and a score threshold. This folder is the minimal repo (two entry scripts only).

## Setup

```cmd
cd skip_sampling_only_github
pip install -r requirements.txt
```

Python 3.8+. GPU recommended. First run downloads MobileNet and YOLOv8n weights.

## Quick start

**Fixed retention** (~20% of processed frames):

```cmd
python scripts/run_sample.py -i your_video.avi --max-frames 500 --retention 0.2
```

**Adaptive retention** (varies by scene; busy traffic keeps more):

```cmd
python scripts/run_sample_adaptive.py -i your_video.avi --max-frames 500
```

**Debug overlay** (green KEEP / red SKIP on every processed frame):

```cmd
python scripts/run_sample_adaptive.py -i your_video.avi --max-frames 500 --debug-video --open
```

Omit `--max-frames` to process the whole file (slow; long videos use stream I/O to limit RAM).


Typical files:

- `*_sampled.mp4` / `*_adaptive_sampled.mp4` — kept frames only (full resolution)
- `*_sample_manifest.json` / `*_adaptive_manifest.json` — kept indices, scores, tau
- `*_sample_report.json` / `*_adaptive_report.json` — size and retention stats
- `*_debug.mp4` — optional, with `--debug-video`

Default config skips the full input reference clip (`skip_input_clip: true`). Use `--no-input-clip` on `run_sample.py` to write `*_input_Nfr.mp4`.

## Progress

Runs print numbered steps, per-step time, and tqdm bars with ETA (e.g. embed, write). Do not pass `--no-progress` unless you want a silent run.

## Speed presets (CLI)

| Flag | Effect |
|------|--------|
| *(none)* | Default: stream read, FP16 batch embed, sparse YOLO, `frame_stride: 2`, uniform context |
| `--light` | Motion gate: CNN+YOLO only on high-motion frames; fewer embed calls |
| `--turbo` | Stricter fast settings (stride 3, 540p scoring, YOLO every 16 frames) |
| `--quality` | Slow path: all frames in RAM, K-means context, YOLO every frame, diverse selection |

Configs: `config/default.yaml`, `config/light.yaml`, `config/quality.yaml`, `config/turbo.yaml`.

## `run_sample.py` options

| Option | Description |
|--------|-------------|
| `-i` / `--input` | Input video |
| `--max-frames N` | Process first N frames (after stride) |
| `--retention 0.2` | Target keep ratio (e.g. 0.2 = 20%) |
| `--adaptive` | Same retention mode as `run_sample_adaptive.py` |
| `--debug-video` | Write labeled debug MP4 |
| `--open` | Open outputs when finished |
| `--no-input-clip` | Skip writing input reference clip |
| `--config path` | Custom YAML |
| `--quality` / `--light` / `--turbo` | Presets (see above) |

## `run_sample_adaptive.py` options

| Option | Description |
|--------|-------------|
| `--target-retention 0.35` | Soft floor when scores are flat (not a fixed keep ratio) |
| `--min-retention` / `--max-retention` | Hard bounds (defaults 12%–70% in yaml) |
| `--tau` | Fixed score threshold instead of adaptive tau |
| Same as sample | `--light`, `--turbo`, `--quality`, `--debug-video`, `--open`, `--max-frames` |

**Adaptive vs fixed:** retention changes with content. Empty scenes skip more; busy moving traffic keeps more (`adaptive.traffic_aware` in yaml). Check `traffic_busy_fraction` in the report.

## Algorithm modes (`scoring.algorithm` in yaml)

| Value | Behavior |
|-------|----------|
| `batch` | Score every processed frame (default) |
| `motion_gated` | Motion pass, then semantic only on ~35% busiest frames (`--light`) |
| `online` | One pass, keep when score > tau after warmup |

Context: `context.mode: uniform` (fast) or `diverse` (K-means, `--quality`).


