import argparse, yaml, joblib
from pathlib import Path
from .pipeline import fit_transform_stream
from .io import save_labels

def main():
    ap = argparse.ArgumentParser(description="Spotify Genre Segmentation pipeline")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model-out", default="models/pipeline.joblib")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    km, scaler, pca, labels_df, sil = fit_transform_stream(cfg)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kmeans": km, "scaler": scaler, "pca": pca, "config": cfg}, args.model_out)
    save_labels(cfg["data"]["output_labels"], labels_df)

    print(f"Saved labels -> {cfg['data']['output_labels']}")
    print(f"Saved model  -> {args.model_out}")
    if sil is not None:
        print(f"Silhouette  -> {sil:.3f}")

if __name__ == "__main__":
    main()
