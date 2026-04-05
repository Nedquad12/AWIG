import numpy as np
import pandas as pd


def score_vsa(df: pd.DataFrame) -> int:

    if len(df) < 30:
        return 0

    vol   = df["volume"].values
    avg7  = float(np.mean(vol[-7:]))
    avg30 = float(np.mean(vol[-30:]))

    if avg30 == 0:
        return 0

    ratio = avg7 / avg30

    if ratio >= 2.0:
        return 2
    elif ratio > 1.0:
        return 1
    elif ratio == 1.0:
        return 0
    elif (1.0 / ratio) >= 2.0:  
        return -2
    else:
        return -1

def analyze(symbol: str, interval: str = "1d", limit: int = 210) -> dict:
    """Fetch data Binance lalu return skor VSA."""
    from indicators.binance_fetcher import get_df
    df = get_df(symbol, interval, limit)
    return {"symbol": symbol.upper(), "interval": interval, "score": score_vsa(df), "df": df}
