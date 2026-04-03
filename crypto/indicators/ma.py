import pandas as pd


MA_PERIODS = [20, 60, 120, 200]


def score_ma(df: pd.DataFrame) -> int:
    if len(df) < max(MA_PERIODS):
        return 0

    price  = float(df["close"].iloc[-1])
    closes = df["close"]

    above_count = sum(
        1 for period in MA_PERIODS
        if price > float(closes.tail(period).mean())
    )

    score_map = {4: 2, 3: 1, 2: 0, 1: -1, 0: -2}
    return score_map[above_count]

def analyze(symbol: str, interval: str = "1d", limit: int = 210) -> dict:
    """Fetch data Binance lalu return skor MA."""
    from indicators.binance_fetcher import get_df
    df = get_df(symbol, interval, limit)
    return {"symbol": symbol.upper(), "interval": interval, "score": score_ma(df), "df": df}
