# src/spotify_segmentation/pipeline.py
import pandas as pd
import numpy as np
from .io import read_in_chunks, save_labels
from .preprocess import clean_and_select, make_scaler
from .features import make_pca
from .cluster import make_clustering, score_clusters

def fit_transform_stream(cfg, sample_chunks=3):
    """
    2-phase streaming pipeline:
    1) take 'sample_chunks' first chunks to fit scaler and PCA (a small sample).
    2) stream over the dataset and call km.partial_fit(X_chunk) so we never keep the whole X in memory.
    Returns trained km, scaler, pca, labels_df, silhouette_estimate (if possible)
    """
    cols = cfg["features"]["use_columns"]
    path = cfg["data"]["input_path"]
    chunk_args = dict(chunksize=cfg["compute"]["chunksize"], downcast=cfg["compute"]["downcast"], usecols=cols + ["track_id"])

    # --- Phase 0: create objects
    scaler = make_scaler() if cfg["features"]["scale"] else None
    pca = make_pca(cfg["features"]["pca_components"])
    km = make_clustering(**cfg["clustering"])

    # --- Phase 1: build a small sample to fit scaler/pca
    sample_frames = []
    sample_ids = []
    cnt = 0
    for chunk in read_in_chunks(path, **chunk_args):
        sample_ids.append(chunk["track_id"].reset_index(drop=True))
        X = clean_and_select(chunk, cols, drop_na=cfg["features"]["drop_na"]).values
        sample_frames.append(X)
        cnt += 1
        if cnt >= sample_chunks:
            break

    if len(sample_frames) == 0:
        raise RuntimeError("No data found in input path or use_columns mismatch.")

    sample_X = np.vstack(sample_frames)
    # fit scaler/pca on the sample
    if scaler is not None:
        scaler.fit(sample_X)
        sample_X = scaler.transform(sample_X)
    if pca is not None:
        pca.fit(sample_X)

    # --- Phase 2: streaming partial_fit over the whole dataset
    labels_parts = []
    ids_parts = []
    # Recreate the chunk generator
    for chunk in read_in_chunks(path, **chunk_args):
        ids_parts.append(chunk["track_id"].reset_index(drop=True))
        X = clean_and_select(chunk, cols, drop_na=cfg["features"]["drop_na"]).values
        if scaler is not None:
            X = scaler.transform(X)
        if pca is not None:
            X = pca.transform(X)

        # If estimator supports partial_fit (MiniBatchKMeans), use it to avoid keeping X_all
        if hasattr(km, "partial_fit"):
            # partial_fit expects 2D array; MiniBatchKMeans partial_fit will update centroids
            km.partial_fit(X)
            # we cannot predict final labels until after final pass — collect predictions after full fit
            # For now store transformed X for later predict (or predict in a second pass)
            labels_parts.append(None)  # placeholder
        else:
            # fallback: accumulate for single-fit
            labels_parts.append(X)

    # If km had no partial_fit we need to fit on stacked data
    if not hasattr(km, "partial_fit"):
        X_all = np.vstack(labels_parts)
        km.fit(X_all)
        labels_all = km.predict(X_all)
    else:
        # We did streaming partial_fit above; now do a second pass to predict labels
        labels_out = []
        for chunk in read_in_chunks(path, **chunk_args):
            X = clean_and_select(chunk, cols, drop_na=cfg["features"]["drop_na"]).values
            if scaler is not None:
                X = scaler.transform(X)
            if pca is not None:
                X = pca.transform(X)
            labels_out.append(km.predict(X))
        labels_all = np.concatenate(labels_out)

    # build ids_all
    ids_all = pd.concat(ids_parts, ignore_index=True)
    out = pd.DataFrame({"track_id": ids_all, "label": labels_all})

    # Try a silhouette estimate on a subsample (if feasible)
    try:
        # Need numeric X for silhouette — sample a small subset by reloading sample_X or re-reading a sample
        # We'll reuse sample_X (fitted earlier) as a small representative set:
        sample_transformed = sample_X.copy()
        labels_sample = km.predict(sample_transformed)
        sil = score_clusters(sample_transformed, labels_sample)
    except Exception:
        sil = None

    return km, scaler, pca, out, sil
