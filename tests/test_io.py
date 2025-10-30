import pandas as pd
from spotify_segmentation.io import _downcast

def test_downcast_basic():
    df = pd.DataFrame({"a":[1.0,2.0], "b":[1,2]})
    df2 = _downcast(df.copy())
    assert df2["a"].dtype.name.startswith("float")
    assert df2["b"].dtype.name.startswith("int")