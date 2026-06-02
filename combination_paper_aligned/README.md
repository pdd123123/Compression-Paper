

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



## Layout

```
combination_paper_aligned/
  config/          skip_default.yaml, compress.yaml (paper_aligned)
  skip_src/
  compress_src/
  scripts/         run_combo_adaptive.py, run_combo_fixed.py
  outputs/
```
