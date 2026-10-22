# VitalVision

A real-time, contactless heart-rate monitor that estimates pulse from ordinary webcam video using remote photoplethysmography (rPPG), built with OpenCV, MediaPipe, and SciPy.

Status: Phase 7 - engineering finish (all phases implemented)

## Setup

```
pip install -e ".[dev]"
scripts/download_model.sh
python scripts/check_env.py
```

## Scripts

Each phase script is runnable on its own from the repo root.

| Script | Phase | What it does |
| --- | --- | --- |
| `scripts/phase1_face_roi.py` | 1 | Live face landmarks and a forehead ROI overlay. |
| `scripts/phase2_signal_extraction.py` | 2 | Records the forehead ROI's mean RGB per frame to `data/raw/signal_<id>.csv`. |
| `scripts/phase3_bpm_estimate.py` | 3 | Offline BPM estimate from a recorded signal CSV (`--plot` to see the spectrum). |
| `scripts/phase4_realtime_pipeline.py` | 4 | Live BPM overlay, re-estimated every second from a rolling 10s buffer. |
| `scripts/phase5_evaluate.py` | 5 | Compares Green / CHROM / POS methods, optionally against a pulse-oximeter ground-truth CSV. |
| `scripts/phase6_demo.py` | 6 | Demo app: switchable method (g/c/p), BPM sparkline, saved session log. |

## Package

Shared logic lives in `src/vitalvision/`:

- `face.py` - MediaPipe Face Landmarker setup, landmark -> forehead ROI polygon, ROI mean colour.
- `signal.py` - uniform resampling, detrending, band-pass filtering, FFT-peak BPM estimation.
- `methods.py` - the three rPPG methods (Green, CHROM, POS) that turn an RGB trace into a pulse signal.
- `pipeline.py` - rolling-buffer BPM estimation combining the above for live use.
- `io.py` - CSV I/O for recorded signals, session logs, and ground-truth readings.
- `evaluate.py` - MAE/RMSE error metrics.

## Tests

```
pytest
```

Tests cover the ROI geometry, the signal pipeline, each rPPG method, the live rolling-buffer estimator, and the evaluation metrics, all against synthetic signals of known BPM.
