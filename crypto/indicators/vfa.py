"""
indicators/vfa.py — Volume Frequency Analysis (VFA)

Hitung rata-rata % change harian volume dan freq (transactions) selama 7 hari bursa.

avg_vol_change  = rata-rata % perubahan volume hari ke hari (6 selisih dari 7 hari)
avg_freq_change = rata-rata % perubahan transactions hari ke hari

Scoring:
  Keduanya negatif                    → -3  (pasar sepi, override semua)
  vol_change >= 2x freq_change        → -1  (volume dominan tapi divergen)
  vol_change > freq_change            → +1
  freq_change >= 2x vol_change        → +3  (frekuensi sangat dominan = retail aktif)
  freq_change > vol_change            → +2
  Jika salah satu negatif:
    → pakai logic di atas hanya dari nilai yang positif
"""

import numpy as np
import pandas as pd


def _avg_pct_change(arr: np.ndarray, n: int = 7) -> float:
    """
    Hitung rata-rata % perubahan harian dari n hari terakhir.
    Menghasilkan n-1 selisih, lalu dirata-rata.
    Return 0.0 jika data tidak cukup atau semua nilai nol.
    """
    tail = arr[-(n + 1):]   # ambil n+1 baris untuk dapat n selisih
    if len(tail) < 2:
        return 0.0

    changes = []
    for i in range(1, len(tail)):
        prev = tail[i - 1]
        if prev == 0:
            continue
        changes.append((tail[i] - prev) / prev * 100.0)

    if not changes:
        return 0.0

    return float(np.mean(changes))


def score_vfa(df: pd.DataFrame) -> int:
    """
    Hitung skor VFA dari data volume dan transactions.

    Args:
        df: DataFrame dengan kolom 'volume' dan 'transactions', diurutkan ascending

    Returns:
        Skor integer antara -3 dan +3
        0 jika data tidak cukup atau kolom transactions tidak ada
    """
    if "transactions" not in df.columns:
        return 0

    if len(df) < 8:   # butuh minimal 8 baris untuk 7 hari + 1 prev
        return 0

    vol_arr  = df["volume"].values
    freq_arr = df["transactions"].values

    avg_vol  = _avg_pct_change(vol_arr)
    avg_freq = _avg_pct_change(freq_arr)

    # Override: keduanya negatif → pasar sepi
    if avg_vol < 0 and avg_freq < 0:
        return -3

    # Salah satu negatif → gunakan hanya yang positif
    if avg_vol < 0:
        # hanya freq yang relevan
        return 2   # freq positif tapi vol negatif → kondisi moderat
    if avg_freq < 0:
        # hanya vol yang relevan
        return 1   # vol positif tapi freq negatif → kondisi lemah

    # Keduanya positif → logic penuh
    if avg_vol == 0 and avg_freq == 0:
        return 0

    # Hindari division by zero
    if avg_freq == 0:
        return -1   # vol ada tapi freq nol → divergen
    if avg_vol == 0:
        return 2    # freq ada tapi vol nol → freq dominan

    vol_to_freq = avg_vol / avg_freq
    freq_to_vol = avg_freq / avg_vol

    if vol_to_freq >= 2.0:
        return -1
    elif avg_vol > avg_freq:
        return 1
    elif freq_to_vol >= 2.0:
        return 3
    else:
        return 2


def get_vfa_detail(df: pd.DataFrame) -> dict:
    """
    Return detail VFA untuk keperluan tampilan tabel /vfa.

    Returns:
        dict: avg_vol_change, avg_freq_change, score
    """
    if "transactions" not in df.columns or len(df) < 8:
        return {"avg_vol": 0.0, "avg_freq": 0.0, "score": 0}

    vol_arr  = df["volume"].values
    freq_arr = df["transactions"].values

    avg_vol  = _avg_pct_change(vol_arr)
    avg_freq = _avg_pct_change(freq_arr)
    score    = score_vfa(df)

    return {
        "avg_vol":  round(avg_vol,  2),
        "avg_freq": round(avg_freq, 2),
        "score":    score,
    }
