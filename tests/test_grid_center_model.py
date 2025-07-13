import math
import pandas as pd
from crypto_bot import grid_center_model


def test_predict_centre_returns_nan():
    df = pd.DataFrame({"close": [1, 2, 3]})
    result = grid_center_model.predict_centre(df)
    assert math.isnan(result)
