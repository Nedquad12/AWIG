

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from typing import Optional

import numpy as np
import pandas as pd

from cache import load as cache_load, list_cached

logger = logging.getLogger(__name__)

# ── Konstanta ─────────────────────────────────────────────────────────────────
MA_PERIODS     = [3, 5, 10, 20]
VT_THRESHOLD   = 5.0    # jarak maksimal untuk VT (%)
T_MIN          = 5.0    # jarak minimal untuk T (%)
T_MAX          = 15.0    # jarak maksimal untuk T (%)
MIN_VALUE_B    = 0.0    # minimum nilai transaksi (miliar)


# ── Helper ────────────────────────────────────────────────────────────────────

def _calc_mas(df: pd.DataFrame) -> Optional[dict]:
    """
    Hitung MA 3/5/10/20 dari kolom close DataFrame.
    Return dict {period: ma_value} atau None jika data tidak cukup.
    """
    closes = df["close"].values
    if len(closes) < max(MA_PERIODS):
        return None

    mas = {}
    for p in MA_PERIODS:
        mas[p] = float(np.mean(closes[-p:]))
    return mas


def _pct_distance(price: float, ma: float) -> float:
    """Persentase jarak harga dari MA."""
    if ma == 0 or np.isnan(ma):
        return float("inf")
    return ((price - ma) / ma) * 100


def _analyze_ticker(ticker: str) -> Optional[dict]:
    """
    Analisis satu ticker dari cache.
    Return dict hasil atau None jika data tidak memenuhi syarat.
    """
    df = cache_load(ticker)
    if df is None or len(df) < max(MA_PERIODS):
        return None

    close  = float(df["close"].iloc[-1])
    volume = float(df["volume"].iloc[-1])
    mas    = _calc_mas(df)

    if mas is None:
        return None

    # Semua MA harus valid
    if any(np.isnan(v) or v == 0 for v in mas.values()):
        return None

    # Harga harus di atas semua MA
    if not all(close > mas[p] for p in MA_PERIODS):
        return None

    # Hitung jarak ke tiap MA
    distances = {p: _pct_distance(close, mas[p]) for p in MA_PERIODS}
    max_dist  = max(distances.values())

    # Filter nilai transaksi
    value_b = (close * volume) / 1_000_000_000
    if value_b < MIN_VALUE_B:
        return None

    return {
        "ticker":    ticker,
        "close":     close,
        "ma20":      mas[20],
        "volume":    volume,
        "value":     value_b,
        "max_dist":  max_dist,
        "distances": distances,
    }


# ── Public: scan semua cache ───────────────────────────────────────────────────

def scan_tight() -> tuple[list[dict], list[dict]]:

    tickers = list_cached()
    vt_list = []
    t_list  = []

    for ticker in tickers:
        try:
            data = _analyze_ticker(ticker)
            if data is None:
                continue

            max_dist = data["max_dist"]

            if max_dist < VT_THRESHOLD:
                vt_list.append(data)
            elif T_MIN <= max_dist < T_MAX:
                t_list.append(data)

        except Exception as e:
            logger.error(f"[{ticker}] Error tight scan: {e}")

    vt_list.sort(key=lambda x: x["value"], reverse=True)
    t_list.sort(key=lambda x: x["value"], reverse=True)

    return vt_list, t_list


# ── Public: skor per ticker ────────────────────────────────────────────────────

def score_tight(ticker: str, vt_set: set, t_set: set) -> int:

    in_vt = ticker in vt_set
    in_t  = ticker in t_set

    if in_vt and in_t:
        return int(3)
    elif in_vt:
        return int(2)
    elif in_t:
        return int(1)
    else:
        return int(0)


# ── Public: format output Telegram ────────────────────────────────────────────

def _format_table(title: str, results: list[dict]) -> str:
    if not results:
        return f"Tidak ada saham yang memenuhi kriteria {title}"

    msg  = "```\n"
    msg += f"{title}\n\n"

    for s in results:
        vol_m = s["volume"] / 1_000_000
        vol_s = f"{vol_m:.0f}" if vol_m >= 1 else f"{vol_m:.1f}"
        val_s = f"{s['value']:.1f}"
        msg += f"{s['ticker']:<6} {s['close']:>8.0f} {s['ma20']:>8.0f} {vol_s:>8} {val_s:>7}\n"

    msg += f"\nTotal: {len(results)} saham\n"
    msg += "Vol=Jutalembar, Val=Miliar USD (close×vol)\n"
    msg += "```"
    return msg


def format_vt(vt_list: list[dict]) -> str:
    return _format_table("🔥 Very Tight Stocks (US)", vt_list)


def format_t(t_list: list[dict]) -> str:
    return _format_table("✨ Tight Stocks (US)", t_list)
