"""
indicators/sr.py — Support & Resistance detector
Terjemahan dari Pine Script "Support & Resistance Pro Toolkit [LuxAlgo]"

Mendukung 4 detection method:
  - Donchian  (default)
  - Pivots
  - CSID
  - ZigZag

Output: list of SRLevel dataclass, siap dipakai untuk charting.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    SR_METHOD_DONCHIAN, SR_METHOD_PIVOTS, SR_METHOD_CSID, SR_METHOD_ZIGZAG,
    SR_SENSITIVITY, SR_ATR_MULT, SR_ATR_PERIOD, SR_MAX_LEVELS,
)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class SRLevel:
    top:            float
    btm:            float
    base_price:     float
    start_bar:      int
    mitigation_bar: int   = -1
    is_support:     bool  = True
    is_mitigated:   bool  = False
    entries:        int   = 0
    strength:       int   = 0
    sweeps:         int   = 0
    traded_volume:  float = 0.0


# ── ATR helper ────────────────────────────────────────────────────────────────

def _calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR, sama seperti ta.atr() di Pine Script."""
    n    = len(close)
    atr  = np.zeros(n)
    tr   = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i]  - close[i-1]))
    # seed
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    # fill leading NaN with cumulative range fallback
    cum = np.cumsum(np.abs(high - low))
    for i in range(period-1):
        atr[i] = cum[i] / (i + 1)
    return atr


# ── Swing detectors (return list of (bar_idx, price, is_high)) ───────────────

def _detect_pivots(high, low, sens: int):
    swings = []
    for i in range(sens, len(high) - sens):
        # pivot high
        if high[i] == max(high[i-sens:i+sens+1]):
            swings.append((i, float(high[i]), True))
        # pivot low
        if low[i] == min(low[i-sens:i+sens+1]):
            swings.append((i, float(low[i]), False))
    return swings


def _detect_donchian(high, low, sens: int):
    """Alternating highest/lowest breakout → swing point."""
    swings = []
    os_dir  = 0
    val     = math.nan
    val_loc = 0

    for i in range(len(high)):
        dh = max(high[max(0, i-sens+1):i+1])
        dl = min(low[max(0, i-sens+1):i+1])
        prev_dh = max(high[max(0, i-sens):i]) if i > 0 else dh
        prev_dl = min(low[max(0, i-sens):i])  if i > 0 else dl

        new_os = 1 if dh > prev_dh else (-1 if dl < prev_dl else os_dir)
        if new_os != os_dir:
            if new_os == 1:
                if not math.isnan(val):
                    swings.append((val_loc, float(val), False))  # swing low
                val = high[i]; val_loc = i
            else:
                if not math.isnan(val):
                    swings.append((val_loc, float(val), True))   # swing high
                val = low[i]; val_loc = i
            os_dir = new_os
        else:
            if os_dir == 1 and high[i] >= val:
                val = high[i]; val_loc = i
            elif os_dir == -1 and low[i] <= val:
                val = low[i]; val_loc = i

    return swings


def _detect_csid(high, low, close, open_, sens: int):
    """Consecutive same-direction candles → swing."""
    swings    = []
    bull_cnt  = 0
    bear_cnt  = 0
    for i in range(len(close)):
        window_h = high[max(0, i-sens+1):i+1]
        window_l = low[max(0, i-sens+1):i+1]

        bull_cnt = bull_cnt + 1 if close[i] > open_[i] else 0
        bear_cnt = bear_cnt + 1 if close[i] < open_[i] else 0

        if bull_cnt >= sens and len(window_h):
            peak_bar = max(range(len(window_h)), key=lambda x: window_h[x])
            swings.append((i - sens + 1 + peak_bar, float(window_h[peak_bar]), True))
            bull_cnt = 0

        if bear_cnt >= sens and len(window_l):
            trough_bar = min(range(len(window_l)), key=lambda x: window_l[x])
            swings.append((i - sens + 1 + trough_bar, float(window_l[trough_bar]), False))
            bear_cnt = 0

    return swings


def _detect_zigzag(high, low, close, deviation_pct: float):
    """ZigZag berdasarkan % deviation dari close."""
    swings  = []
    zz_high = math.nan; zz_low = math.nan
    zz_hbar = 0;        zz_lbar = 0
    zz_dir  = 0

    for i in range(len(close)):
        dev = close[i] * deviation_pct / 100.0
        if zz_dir == 0:
            zz_high = high[i]; zz_low = low[i]
            zz_hbar = i;       zz_lbar = i
            zz_dir  = 1 if close[i] > (open if hasattr(open, '__len__') else close[i]) else -1
            if zz_dir == 0:
                zz_dir = 1
        elif zz_dir == 1:
            if high[i] > zz_high:
                zz_high = high[i]; zz_hbar = i
            elif low[i] < zz_high - dev:
                swings.append((zz_hbar, float(zz_high), True))
                zz_low = low[i]; zz_lbar = i; zz_dir = -1
        else:
            if low[i] < zz_low:
                zz_low = low[i]; zz_lbar = i
            elif high[i] > zz_low + dev:
                swings.append((zz_lbar, float(zz_low), False))
                zz_high = high[i]; zz_hbar = i; zz_dir = 1

    return swings


# ── Overlap handling ──────────────────────────────────────────────────────────

def _handle_overlap(levels: list[SRLevel], top: float, btm: float,
                    base: float, start: int, is_sup: bool) -> bool:
    """
    Sama seperti handle_structure() di Pine Script.
    Mode: Hide Overlapping (Oldest Precedence) — cocok untuk tampilan bersih.
    Jika overlap dengan level yang sudah ada → skip level baru.
    """
    for lv in levels:
        if not lv.is_mitigated and lv.is_support == is_sup:
            overlap = max(btm, lv.btm) < min(top, lv.top)
            if overlap:
                return False   # skip — oldest precedence
    levels.insert(0, SRLevel(top=top, btm=btm, base_price=base,
                              start_bar=start, is_support=is_sup))
    return True


# ── Mitigation & stats update ─────────────────────────────────────────────────

def _update_mitigation(levels: list[SRLevel],
                       high: np.ndarray, low: np.ndarray,
                       close: np.ndarray, volume: np.ndarray,
                       open_: np.ndarray):
    """Loop semua bar, update entries/sweeps/volume dan tandai mitigation."""
    n = len(close)
    for i in range(n):
        for lv in levels:
            if lv.is_mitigated:
                continue
            # mitigation check
            if lv.is_support and close[i] < lv.btm:
                lv.is_mitigated   = True
                lv.mitigation_bar = i
                continue
            if not lv.is_support and close[i] > lv.top:
                lv.is_mitigated   = True
                lv.mitigation_bar = i
                continue

            # only count from start_bar onwards
            if i < lv.start_bar:
                continue

            # traded volume inside zone
            if high[i] >= lv.btm and low[i] <= lv.top:
                lv.traded_volume += float(volume[i])

            if lv.is_support:
                if low[i] <= lv.top and close[i] >= lv.btm:
                    lv.entries += 1
                if low[i] < lv.btm and min(close[i], open_[i]) > lv.btm:
                    lv.sweeps += 1
            else:
                if high[i] >= lv.btm and close[i] <= lv.top:
                    lv.entries += 1
                if high[i] > lv.top and max(close[i], open_[i]) < lv.top:
                    lv.sweeps += 1


# ── Main public function ──────────────────────────────────────────────────────

def detect_sr(
    df:     pd.DataFrame,
    method: str   = SR_METHOD_DONCHIAN,
    sens:   float = SR_SENSITIVITY,
    atr_mult: float = SR_ATR_MULT,
    max_levels: int = SR_MAX_LEVELS,
) -> tuple[list[SRLevel], list[SRLevel]]:
    """
    Deteksi level Support & Resistance dari DataFrame OHLCV.

    Args:
        df         : DataFrame dengan kolom open/high/low/close/volume
        method     : "Donchian" | "Pivots" | "CSID" | "ZigZag"
        sens       : sensitivity (lookback untuk Pivot/Donchian/CSID,
                     deviation% untuk ZigZag)
        atr_mult   : kedalaman zone (ATR multiplier)
        max_levels : max active levels yang dikembalikan

    Returns:
        (active_levels, broken_levels)
        Masing-masing list[SRLevel], sudah difilter max_levels.
    """
    high   = df["high"].values
    low    = df["low"].values
    close  = df["close"].values
    open_  = df["open"].values
    volume = df["volume"].values
    sens_i = max(1, int(sens))

    atr = _calc_atr(high, low, close, SR_ATR_PERIOD)

    # ── Detect swings ──────────────────────────────────────────────────────
    if method == SR_METHOD_PIVOTS:
        swings = _detect_pivots(high, low, sens_i)
    elif method == SR_METHOD_CSID:
        swings = _detect_csid(high, low, close, open_, sens_i)
    elif method == SR_METHOD_ZIGZAG:
        swings = _detect_zigzag(high, low, close, float(sens))
    else:  # Donchian default
        swings = _detect_donchian(high, low, sens_i)

    # ── Build levels ───────────────────────────────────────────────────────
    levels: list[SRLevel] = []
    for bar_i, price, is_high in swings:
        cur_atr = float(atr[min(bar_i, len(atr)-1)])
        if is_high:   # resistance
            top = price + cur_atr * 0.0   # no breakout buffer
            btm = price - cur_atr * atr_mult
            _handle_overlap(levels, top, btm, price, bar_i, is_sup=False)
        else:         # support
            top = price + cur_atr * atr_mult
            btm = price - cur_atr * 0.0
            _handle_overlap(levels, top, btm, price, bar_i, is_sup=True)

    # ── Trim array gar tidak terlalu besar ─────────────────────────────────
    if len(levels) > 200:
        levels = levels[:200]

    # ── Update mitigation & stats ──────────────────────────────────────────
    _update_mitigation(levels, high, low, close, volume, open_)

    # ── Split active vs broken ─────────────────────────────────────────────
    active_sup = [l for l in levels if not l.is_mitigated and     l.is_support][:max_levels]
    active_res = [l for l in levels if not l.is_mitigated and not l.is_support][:max_levels]
    broken     = [l for l in levels if     l.is_mitigated]

    return active_sup + active_res, broken
