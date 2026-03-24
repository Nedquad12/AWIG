"""
scorer.py — Memanggil semua indikator dan menghasilkan skor final per saham

Total score dihitung dengan weight per fitur dari weight_manager.
Jika tidak ada weight file → pakai default (semua 1.0, total = sum biasa).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

import pandas as pd

from indicators import (
    score_vsa, score_rsi, score_macd, score_ma,
    calculate_ip, score_ip, score_fsa, score_vfa, score_wcc,
    score_srst,
)
from weight_manager import load_weights, apply_weights


def calculate_all_scores(ticker: str, df: pd.DataFrame, tight_score: int = 0) -> dict:
    """
    Hitung semua skor untuk satu saham dari DataFrame OHLCV.

    Args:
        ticker       : kode saham (e.g. "AAPL")
        df           : DataFrame dengan kolom date/open/high/low/close/volume/transactions
        tight_score  : skor VT/T dari tight.py, default 0 jika belum dihitung

    Returns:
        Dict berisi semua skor dan metadata
    """
    vsa    = score_vsa(df)
    rsi    = score_rsi(df)
    macd   = score_macd(df)
    ma     = score_ma(df)
    ip_raw = calculate_ip(df)
    ip_pts = score_ip(ip_raw)
    fsa    = score_fsa(df)
    vfa    = score_vfa(df)
    wcc    = score_wcc(df)
    srst   = score_srst(df)

    # Kumpulkan semua skor per fitur
    scores = {
        "vsa":      vsa,
        "fsa":      fsa,
        "vfa":      vfa,
        "wcc":      wcc,
        "srst":     srst,
        "rsi":      rsi,
        "macd":     macd,
        "ma":       ma,
        "ip_score": ip_pts,
        "tight":    tight_score,
    }

    # Load weight per ticker (fallback ke default 1.0 jika tidak ada)
    weights = load_weights(ticker)
    total   = apply_weights(scores, weights)

    price  = float(df["close"].iloc[-1])
    prev   = float(df["close"].iloc[-2]) if len(df) > 1 else price
    change = ((price - prev) / prev * 100) if prev != 0 else 0.0

    return {
        "ticker":   ticker,
        "date":     datetime.today().strftime("%Y-%m-%d"),
        "price":    round(price,  2),
        "change":   round(change, 2),
        # skor per indikator (raw, sebelum weight)
        "vsa":      vsa,
        "fsa":      fsa,
        "vfa":      vfa,
        "wcc":      wcc,
        "srst":     srst,
        "rsi":      rsi,
        "macd":     macd,
        "ma":       ma,
        "ip_raw":   round(ip_raw, 2),
        "ip_score": ip_pts,
        "tight":    tight_score,
        # total dengan weight
        "total":    round(total, 4),
    }
