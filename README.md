# Spotify Genre Segmentation — Streaming Pipeline (feat/pipeline-refactor) 🚀

> Lightweight, production-style refactor: YAML-configured, CLI-driven pipeline that streams Spotify audio features, preprocesses them, and trains a scalable `MiniBatchKMeans` model via `partial_fit`. Outputs `data/processed/track_labels.csv` and `models/pipeline.joblib`.

## Quickstart (short)
```bash
git clone https://github.com/thisisyashsharma/ML-Projects.git
cd ML-Projects
git fetch origin
git checkout feat/pipeline-refactor
python -m venv .venv && source .venv/bin/activate  # or PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spotify_segmentation.cli --config config/default.yaml --model-out models/pipeline.joblib
# or offline: python run_pipeline.py --config config/default.yaml --model-out models/pipeline.joblib
```

## Core ideas
- **Reproducible:** all fields in `config/default.yaml`.  
- **Memory-efficient:** chunked CSV reads + dtype downcasting.  
- **Scalable:** `MiniBatchKMeans` using streaming `partial_fit`.  
- **Dev-friendly:** CLI + `run_pipeline.py`, basic unit tests, CI workflow.

## Config highlights
Edit `config/default.yaml`:
- `data.input_path` — CSV path  
- `features.use_columns` — feature list  
- `clustering.n_clusters`, `batch_size` — tuning knobs  
- `compute.chunksize`, `downcast` — memory controls

## How it runs (brief)
1. Fit transforms on a few initial chunks (scaler ± PCA).  
2. Stream training with `partial_fit` per chunk.  
3. Second pass: predict labels chunk-by-chunk and save CSV + joblib artifact.

## Quick tips
- Sweep `n_clusters` (8–40) and compare silhouette.  
- One-hot or drop `key`/`mode` for better Euclidean clustering.  
- Use PCA (e.g., 10 components) to denoise before clustering.

## Tests & CI
- Tests: `tests/` — run with `pytest -q`.  
- CI: GitHub Actions runs tests on push/PR.

## Outputs
- `data/processed/track_labels.csv` (track_id,label)  
- `models/pipeline.joblib` ({kmeans, scaler, pca, config})

## Contact
Maintainer: **Yash Sharma** — https://github.com/thisisyashsharma

