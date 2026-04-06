
import pandas as pd


def score_wcc(df: pd.DataFrame) -> int:

    if len(df) < 2:
        return 0

    today = df.iloc[-1]
    prev  = df.iloc[-2]

    o = float(today["open"])
    h = float(today["high"])
    l = float(today["low"])
    c = float(today["close"])
    c_prev = float(prev["close"])

    if o == 0:
        return 0

    if c == c_prev:
        return 0

    close_up = c > c_prev  

    open_to_close = (c - o) / o * 100

    if close_up and open_to_close < 0:
        close_up = False
    elif not close_up and open_to_close > 0:
        close_up = True

    if open_to_close == 0:
        return 0

    if close_up:
        # WCC Up
        low_to_open = (o - l) / o * 100
        ratio = (low_to_open / open_to_close) * 100
        
        if ratio >= 650:
            return 3
        elif ratio >= 350:
            return 2
        elif ratio >= 50:
            return 1
        else:
            return 0

    else:
        high_to_open = (h - o) / o * 100
        ratio = (high_to_open / open_to_close) * 100 

        if ratio <= -650:
            return -3
        elif ratio <= -350:
            return -2
        elif ratio <= -50:
            return -1
        else:
            return 0


def get_wcc_detail(df: pd.DataFrame) -> dict:

    if len(df) < 2:
        return {"open_to_close": 0.0, "wick_to_body": 0.0, "ratio": 0.0, "score": 0, "direction": "-"}

    today  = df.iloc[-1]
    prev   = df.iloc[-2]

    o      = float(today["open"])
    h      = float(today["high"])
    l      = float(today["low"])
    c      = float(today["close"])
    c_prev = float(prev["close"])

    if o == 0:
        return {"open_to_close": 0.0, "wick_to_body": 0.0, "ratio": 0.0, "score": 0, "direction": "-"}

    close_up      = c > c_prev
    open_to_close = (c - o) / o * 100
    direction     = "UP" if close_up else ("DOWN" if c != c_prev else "FLAT")

    if c == c_prev or open_to_close == 0:
        return {"open_to_close": 0.0, "wick_to_body": 0.0, "ratio": 0.0, "score": 0, "direction": direction}

    actual_up = close_up
    if close_up and open_to_close < 0:
        actual_up = False
        direction = "DOWN"
    elif not close_up and open_to_close > 0:
        actual_up = True
        direction = "UP"

    if actual_up:
        wick_to_body = (o - l) / o * 100
    else:
        wick_to_body = (h - o) / o * 100

    ratio = (wick_to_body / open_to_close) * 100
    score = score_wcc(df)

    return {
        "open_to_close": round(open_to_close, 2),
        "wick_to_body":  round(wick_to_body,  2),
        "ratio":         round(ratio,          2),
        "score":         score,
        "direction":     direction,
    }

def analyze(symbol: str, interval: str = "1d", limit: int = 210) -> dict:
    """Fetch data Binance lalu return skor + detail WCC."""
    from indicators.binance_fetcher import get_df
    df = get_df(symbol, interval, limit)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "score": score_wcc(df),
        "detail": get_wcc_detail(df),
        "df": df,
    }
