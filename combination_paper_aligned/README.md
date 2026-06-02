# Skip sampling + paper-aligned compression (SkipComp combo)

Same skip pipeline as `combination_simple`, but compression uses **`run_paper_aligned`** settings:
soft-edge (Canny + K-means) in `.seccomp` + H.264 annex playback (CRF 30).

## Quick start

```cmd
cd combination_paper_aligned
pip install -r requirements.txt
```

```cmd
python scripts/run_combo_adaptive.py -i video.avi --max-frames 500
python scripts/run_combo_fixed.py -i video.avi --max-frames 500 --retention 0.2
```

Common flags: `--stem`, `-o outputs`, `--baseline-crf 28`, `--max-frames`, `--final-only`.

## Pipeline

1. Skip sampling — EfficientNet-B0 + YOLOv8s, adaptive v4  
2. Compress — paper soft-edge + H.264 annex (`config/compress.yaml`)  
3. Decompress — `{stem}_recon.mp4` (H.264 playback)

## vs `combination_simple`

| | `combination_simple` | `combination_paper_aligned` |
|--|----------------------|----------------------------|
| Compression | legacy delivery (CRF 26) | paper_aligned (soft-edge + CRF 30) |
| Paper edge in bitstream | partial | yes (validated) |
| Playback quality | good | matches `run_transmit` / `run_paper_aligned` |

## Layout

```
combination_paper_aligned/
  config/          skip_default.yaml, compress.yaml (paper_aligned)
  skip_src/
  compress_src/
  scripts/         run_combo_adaptive.py, run_combo_fixed.py
  outputs/
```
