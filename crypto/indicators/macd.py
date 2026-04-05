import pandas as pd


def _compute_macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float]:
    ema_fast    = closes.ewm(span=fast,   adjust=False).mean()
    ema_slow    = closes.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def score_macd(df: pd.DataFrame) -> int:
    if len(df) < 35:  
        return 0

    macd_val, signal_val = _compute_macd(df["close"])

    score = 0

    if macd_val > signal_val:
        score += 1
    elif macd_val < signal_val:
        score -= 1

    if macd_val > 0:
        score += 1
    elif macd_val < 0:
        score -= 1

    return score

def analyze(symbol: str, interval: str = "1d", limit: int = 210) -> dict:
    """Fetch data Binance lalu return skor MACD."""
    from indicators.binance_fetcher import get_df
    df = get_df(symbol, interval, limit)
    return {"symbol": symbol.upper(), "interval": interval, "score": score_macd(df), "df": df}
