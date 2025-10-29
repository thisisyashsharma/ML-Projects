# Spotify Genre Segmentation

**Goal**: cluster tracks into segments using Spotify audio features at scale.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spotify_segmentation.cli --config config/default.yaml --model-out models/pipeline.joblib
