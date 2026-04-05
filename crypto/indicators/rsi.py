import numpy as np
import pandas as pd


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def score_rsi(df: pd.DataFrame, period: int = 14) -> int:

    if len(df) < period + 1:
        return 0

    rsi = _compute_rsi(df["close"].values, period)

    if rsi > 70:
        return -1
    elif rsi >= 50:
        return 0
    elif rsi >= 30:
        return 1
    else:
        return 2


def analyze(symbol: str, interval: str = "1d", limit: int = 210) -> dict:
    """Fetch data Binance lalu return skor RSI."""
    from indicators.binance_fetcher import get_df
    df = get_df(symbol, interval, limit)
    return {"symbol": symbol.upper(), "interval": interval, "score": score_rsi(df), "df": df}
