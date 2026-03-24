"""
score_history.py — Hitung score history 300 bar dan simpan ke JSON + SQLite

Alur:
  1. Terima df 500 bar dari fetch
  2. Loop bar ke-200 s/d bar ke-499 (300 iterasi)
     Untuk setiap bar i:
       - Ambil df[:i+1] sebagai window data
       - Hitung semua skor (vsa, rsi, macd, ma, ip, fsa, vfa, wcc, srst)
       - Simpan hasil + metadata OHLCV ke list
  3. Simpan list ke JSON  → /home/ec2-user/us/500/{TICKER}.json
  4. Upsert ke SQLite DB  → /home/ec2-user/us/train/train.db

Format JSON:
  {
    "ticker": "AAPL",
    "generated_at": "...",
    "total_bars": 300,
    "data": [
      {
        "date": "2025-01-01",
        "bar_idx": 200,
        "price": 150.0,
        "open": 149.0, "high": 151.0, "low": 148.5,
        "volume": 1234567, "transactions": 45678,
        "change_pct": 0.5,
        "vsa": 1, "fsa": 0, "vfa": 2, "wcc": 1, "srst": 0,
        "rsi": 1, "macd": 1, "ma": 2,
        "ip_raw": 2.33, "ip_score": 1.0,
        "tight": 0,
        "total": 9.0
      },
      ...
    ]
  }
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from config import OHLCV_500_DIR, SCORE_WARMUP, SCORE_BARS
from indicators import (
    score_vsa, score_rsi, score_macd, score_ma,
    calculate_ip, score_ip, score_fsa, score_vfa, score_wcc,
    score_srst,
)
from train_db import upsert_score_rows, init_db

logger = logging.getLogger(__name__)


def _ensure_dirs():
    os.makedirs(OHLCV_500_DIR, exist_ok=True)


def _json_path(ticker: str) -> str:
    return os.path.join(OHLCV_500_DIR, f"{ticker.upper()}.json")


def _score_single_bar(window_df: pd.DataFrame, tight_score: int = 0) -> dict:
    """Hitung semua skor dari window DataFrame (semua bar sampai bar ini)."""
    vsa    = score_vsa(window_df)
    rsi    = score_rsi(window_df)
    macd   = score_macd(window_df)
    ma     = score_ma(window_df)
    ip_raw = calculate_ip(window_df)
    ip_pts = score_ip(ip_raw)
    fsa    = score_fsa(window_df)
    vfa    = score_vfa(window_df)
    wcc    = score_wcc(window_df)
    srst   = score_srst(window_df)

    total  = vsa + rsi + macd + ma + ip_pts + tight_score + fsa + vfa + wcc + srst

    return {
        "vsa":      vsa,
        "fsa":      fsa,
        "vfa":      vfa,
        "wcc":      wcc,
        "srst":     srst,
        "rsi":      rsi,
        "macd":     macd,
        "ma":       ma,
        "ip_raw":   round(ip_raw, 4),
        "ip_score": ip_pts,
        "tight":    tight_score,
        "total":    round(total, 2),
    }


def build_score_history(
    ticker: str,
    df: pd.DataFrame,
    tight_score: int = 0,
) -> list[dict]:
    """
    Hitung score untuk setiap bar dari SCORE_WARMUP hingga bar terakhir.

    Args:
        ticker      : kode saham
        df          : DataFrame 500 bar, diurutkan ascending
        tight_score : tight score dari run /9 (statis, sama untuk semua bar)

    Returns:
        list of dict, panjang = SCORE_BARS (300)
    """
    n = len(df)
    results = []

    # Bar index mulai dari SCORE_WARMUP (200) sampai bar terakhir
    start_idx = SCORE_WARMUP  # index 200 → window df[0:201] = 201 bar

    for i in range(start_idx, n):
        window = df.iloc[: i + 1]   # semua data sampai bar ke-i
        row    = df.iloc[i]

        # Metadata bar
        date_val = row["date"]
        if hasattr(date_val, "strftime"):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)[:10]

        # Harga sebelumnya untuk change_pct
        prev_close = float(df.iloc[i - 1]["close"]) if i > 0 else float(row["close"])
        close      = float(row["close"])
        change_pct = ((close - prev_close) / prev_close * 100) if prev_close != 0 else 0.0

        scores = _score_single_bar(window, tight_score)

        entry = {
            "date":         date_str,
            "bar_idx":      i,
            "price":        round(close, 4),
            "open":         round(float(row["open"]),   4),
            "high":         round(float(row["high"]),   4),
            "low":          round(float(row["low"]),    4),
            "volume":       float(row["volume"]),
            "transactions": int(row.get("transactions", 0)),
            "change_pct":   round(change_pct, 4),
            **scores,
        }
        results.append(entry)

    logger.info(f"[{ticker}] Score history: {len(results)} bar dihitung")
    return results


def save_score_history(ticker: str, history: list[dict]):
    """
    Simpan score history ke JSON file di OHLCV_500_DIR.
    """
    _ensure_dirs()
    payload = {
        "ticker":       ticker.upper(),
        "generated_at": datetime.utcnow().isoformat(),
        "total_bars":   len(history),
        "data":         history,
    }
    path = _json_path(ticker)
    with open(path, "w") as f:
        json.dump(payload, f)
    logger.info(f"[{ticker}] JSON disimpan: {path}")


def load_score_history(ticker: str) -> Optional[list[dict]]:
    """
    Baca score history dari JSON file.

    Returns:
        list of dict atau None jika file tidak ada
    """
    path = _json_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload.get("data", [])
    except Exception as e:
        logger.error(f"[{ticker}] Gagal baca JSON: {e}")
        return None


def process_and_store(
    ticker: str,
    df: pd.DataFrame,
    tight_score: int = 0,
):
    """
    Entry point utama: hitung score history, simpan JSON + DB.

    Args:
        ticker      : kode saham
        df          : DataFrame 500 bar
        tight_score : dari tight.py
    """
    try:
        history = build_score_history(ticker, df, tight_score)
        if not history:
            logger.warning(f"[{ticker}] Tidak ada history yang dihasilkan")
            return

        # Simpan JSON
        save_score_history(ticker, history)

        # Siapkan rows untuk DB (tambahkan ticker)
        db_rows = [{"ticker": ticker.upper(), **h} for h in history]

        # Upsert ke SQLite
        upsert_score_rows(db_rows)

        logger.info(f"[{ticker}] Selesai: {len(history)} bar → JSON + DB")

    except Exception as e:
        logger.error(f"[{ticker}] Error process_and_store: {e}")
