"""
indicators/wcc.py — Wick Candle Change (WCC)

Penentu Up/Down: close hari ini vs close kemarin
  close > close kemarin → WCC Up logic
  close < close kemarin → WCC Down logic
  close == close kemarin → skor 0

WCC Up (open_to_close positif):
  low_to_open   = (open - low) / open * 100
  open_to_close = (close - open) / open * 100
  ratio         = (low_to_open / open_to_close) * 100
  ratio < 50    →  0
  ratio >= 50   → +1
  ratio >= 100  → +2
  ratio >= 150  → +3
  ratio >= 200  → +4

WCC Down (open_to_close negatif):
  high_to_open  = (high - open) / open * 100
  open_to_close = (close - open) / open * 100  ← negatif
  ratio         = (high_to_open / open_to_close) * 100  ← negatif
  ratio > -50   →  0
  ratio <= -50  → -1
  ratio <= -100 → -2
  ratio <= -150 → -3
  ratio <= -200 → -4

Jika arah candle (open_to_close) berlawanan dengan arah close,
logic dibalik ke arah candle yang sebenarnya.
"""

import pandas as pd


def score_wcc(df: pd.DataFrame) -> int:
    """
    Hitung skor WCC dari candle terbaru.

    Args:
        df: DataFrame dengan kolom open/high/low/close, diurutkan ascending

    Returns:
        Skor integer antara -4 dan +4
        0 jika data tidak cukup
    """
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

    # Tentukan arah dari close hari ini vs kemarin
    if c == c_prev:
        return 0

    close_up = c > c_prev  # True = WCC Up, False = WCC Down

    # Hitung open_to_close — ini yang menentukan candle bullish/bearish
    open_to_close = (c - o) / o * 100

    # Jika arah candle berlawanan dengan arah close → pakai logic sebaliknya
    if close_up and open_to_close < 0:
        # Gap up tapi candle bearish → pakai WCC Down logic
        close_up = False
    elif not close_up and open_to_close > 0:
        # Gap down tapi candle bullish → pakai WCC Up logic
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
        # WCC Down
        high_to_open = (h - o) / o * 100
        ratio = (high_to_open / open_to_close) * 100  # negatif / negatif = positif... tunggu

        # open_to_close negatif, high_to_open positif
        # ratio = positif / negatif = negatif ✓

        if ratio <= -650:
            return -3
        elif ratio <= -350:
            return -2
        elif ratio <= -50:
            return -1
        else:
            return 0


def get_wcc_detail(df: pd.DataFrame) -> dict:
    """
    Return detail WCC untuk keperluan tampilan tabel /wcc.

    Returns:
        dict: open_to_close, low_to_open atau high_to_open, ratio, score
    """
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

    # Sesuaikan direction jika candle berlawanan
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
