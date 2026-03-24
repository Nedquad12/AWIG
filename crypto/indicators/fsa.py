"""
indicators/fsa.py — Frekuensi Super Analysis

Membandingkan rata-rata jumlah transaksi 7 hari bursa vs 30 hari bursa.
Data transaksi diambil dari field `n` (number of transactions) pada
endpoint Massive.com aggregates.

Logika:
  avg7 >= 2x avg30  →  +2
  avg7 > avg30       →  +1
  avg7 == avg30      →   0
  avg7 < avg30       →  -1
"""

import numpy as np
import pandas as pd


def score_fsa(df: pd.DataFrame) -> int:
    """
    Bandingkan rata-rata jumlah transaksi 7 hari bursa vs 30 hari bursa.

    Args:
        df: DataFrame dengan kolom 'transactions', diurutkan ascending

    Returns:
        Skor integer antara -1 dan +2
        0 jika kolom 'transactions' tidak tersedia atau data tidak cukup
    """
    if "transactions" not in df.columns:
        return 0

    if len(df) < 30:
        return 0

    txn    = df["transactions"].values
    avg7   = float(np.mean(txn[-7:]))
    avg30  = float(np.mean(txn[-30:]))

    if avg30 == 0:
        return 0

    ratio = avg7 / avg30

    if ratio >= 2.0:
        return 2
    elif ratio > 1.0:
        return 1
    elif ratio == 1.0:
        return 0
    else:
        return -1
