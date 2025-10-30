import pandas as pd
from spotify_segmentation.pipeline import fit_transform_stream

def test_pipeline_small(tmp_path):
    df = pd.DataFrame({
        "track_id":[1,2,3,4],
        "danceability":[0.5,0.6,0.55,0.52],
        "energy":[0.7,0.4,0.6,0.65],
        "valence":[0.2,0.3,0.25,0.22],
        "tempo":[120,130,125,128],
        "loudness":[-5.0,-6.0,-5.5,-5.2],
        "speechiness":[0.03,0.04,0.02,0.03],
        "acousticness":[0.01,0.2,0.05,0.03],
        "instrumentalness":[0,0,0,0],
        "liveness":[0.1,0.2,0.15,0.12],
        "key":[1,2,3,4],
        "mode":[1,1,0,1],
        "duration_ms":[200000,180000,210000,190000]
    })
    csv = tmp_path / "small.csv"
    df.to_csv(csv, index=False)
    cfg = {
      "data":{"input_path": str(csv), "output_labels": str(tmp_path/"out.csv"), "cache_dir": str(tmp_path/"cache"), "use_parquet": False},
      "features":{"use_columns": ["danceability","energy","valence","tempo","loudness","speechiness","acousticness","instrumentalness","liveness","key","mode","duration_ms"], "drop_na": True, "scale": True, "pca_components": 0},
      "clustering":{"method":"minibatchkmeans","n_clusters":2,"batch_size":2,"max_iter":10,"random_state":0},
      "compute":{"parallel_backend":"joblib","n_jobs":1,"chunksize":2,"downcast":True}
    }
    km, scaler, pca, out, sil = fit_transform_stream(cfg)
    assert "track_id" in out.columns and "label" in out.columns