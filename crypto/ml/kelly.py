"""
ml/kelly.py — Kelly Criterion + Monte Carlo position sizing.

Alur lengkap:
  1. Hitung ATR (14 candle) dari raw_df → dapat SL/TP distance
  2. Hitung RR aktual dari SL/TP (bukan placeholder)
  3. Hitung Kelly dari winrate + RR aktual
  4. Monte Carlo validasi fraction → safe_fraction
  5. Hitung leverage dari SL distance + safe_fraction → clamp 3–15

Output:
  entry_price, stop_loss, take_profit, leverage, qty_fraction
  + ringkasan untuk dikirim ke AI sebagai konteks
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kelly
KELLY_MULTIPLIER  = 0.25   # Quarter Kelly
MIN_FRACTION      = 0.005  # 0.5%

# Monte Carlo
MAX_DRAWDOWN_PCT  = 0.20   # P5 drawdown max 20%
MC_SIMULATIONS    = 1000
MC_TRADES         = 100

# ATR
ATR_PERIOD        = 14
ATR_SL_MULTIPLIER = 1.5    # SL = 1.5 × ATR
ATR_TP_MULTIPLIER = 3.0    # TP = 3.0 × ATR → RR = 2.0 base

# Leverage
MIN_LEVERAGE      = 3
MAX_LEVERAGE      = 15

# Defaults
DEFAULT_WINRATE   = 0.50
DEFAULT_RR        = 2.0


# ------------------------------------------------------------------
# ATR calculation
# ------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """
    Hitung Average True Range dari last `period` candle.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    if len(df) < period + 1:
        # Fallback: pakai simple high-low range
        return float(df["high"].tail(period).values - df["low"].tail(period).values).mean()

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    tr = []
    for i in range(1, len(df)):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr.append(max(hl, hpc, lpc))

    return float(np.mean(tr[-period:]))


# ------------------------------------------------------------------
# SL / TP dari ATR
# ------------------------------------------------------------------

def compute_sltp(
    df: pd.DataFrame,
    direction: str,            # "LONG" or "SHORT"
    sl_multiplier: float = ATR_SL_MULTIPLIER,
    tp_multiplier: float = ATR_TP_MULTIPLIER,
) -> dict:
    """
    Hitung entry, SL, TP berdasarkan ATR dan arah trade.

    Returns:
        dict: entry_price, stop_loss, take_profit, atr,
              sl_distance, tp_distance, rr_ratio
    """
    entry = float(df["close"].iloc[-1])
    atr   = _compute_atr(df)

    sl_dist = atr * sl_multiplier
    tp_dist = atr * tp_multiplier
    rr      = tp_dist / sl_dist   # = tp_multiplier / sl_multiplier

    if direction == "LONG":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:  # SHORT
        sl = entry + sl_dist
        tp = entry - tp_dist

    sl_pct = sl_dist / entry   # sebagai % dari entry (untuk leverage calc)

    return {
        "entry_price":  round(entry,   8),
        "stop_loss":    round(sl,      8),
        "take_profit":  round(tp,      8),
        "atr":          round(atr,     8),
        "sl_distance":  round(sl_dist, 8),
        "tp_distance":  round(tp_dist, 8),
        "sl_pct":       round(sl_pct,  6),
        "rr_ratio":     round(rr,      4),
    }


# ------------------------------------------------------------------
# Kelly Criterion
# ------------------------------------------------------------------

def _kelly_full(winrate: float, rr: float) -> float:
    p = winrate
    q = 1.0 - p
    b = rr
    return (p * b - q) / b


# ------------------------------------------------------------------
# Monte Carlo
# ------------------------------------------------------------------

def _run_monte_carlo(
    fraction: float,
    winrate: float,
    rr: float,
    n_simulations: int = MC_SIMULATIONS,
    n_trades: int = MC_TRADES,
    seed: int = 42,
) -> dict:
    """
    Simulasi n_simulations equity curve, masing-masing n_trades trade.
    Return distribusi outcome dan drawdown.
    """
    rng = np.random.default_rng(seed)

    final_equities = np.zeros(n_simulations)
    max_drawdowns  = np.zeros(n_simulations)

    for i in range(n_simulations):
        equity = 1.0
        peak   = 1.0
        max_dd = 0.0

        outcomes = rng.random(n_trades) < winrate

        for win in outcomes:
            if win:
                equity *= (1 + fraction * rr)
            else:
                equity *= (1 - fraction)

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        final_equities[i] = equity
        max_drawdowns[i]  = max_dd

    return {
        "median_final":    round(float(np.median(final_equities)),        4),
        "p5_final":        round(float(np.percentile(final_equities, 5)), 4),
        "p95_final":       round(float(np.percentile(final_equities, 95)),4),
        "max_drawdown_p5": round(float(np.percentile(max_drawdowns, 95)), 4),
        "max_drawdown_med":round(float(np.median(max_drawdowns)),         4),
        "ruin_rate":       round(float(np.mean(final_equities < 0.5)),    4),
        "n_simulations":   n_simulations,
        "n_trades":        n_trades,
    }


def _find_safe_fraction(
    fraction: float,
    winrate: float,
    rr: float,
    max_drawdown: float = MAX_DRAWDOWN_PCT,
    step: float = 0.001,
) -> tuple[float, dict]:
    """Turunkan fraction sampai P5 drawdown <= max_drawdown."""
    f  = fraction
    mc = _run_monte_carlo(f, winrate, rr)

    if mc["max_drawdown_p5"] <= max_drawdown:
        return f, mc

    while f > MIN_FRACTION:
        f  = max(MIN_FRACTION, round(f - step, 6))
        mc = _run_monte_carlo(f, winrate, rr)
        if mc["max_drawdown_p5"] <= max_drawdown:
            break

    return f, mc


# ------------------------------------------------------------------
# Leverage dari SL distance + fraction
# ------------------------------------------------------------------

def _compute_leverage(
    sl_pct: float,
    fraction: float,
    risk_per_trade: float,
) -> int:
    """
    Hitung leverage agar risk per trade = risk_per_trade.

    risk_per_trade = fraction × sl_pct × leverage
    leverage       = risk_per_trade / (fraction × sl_pct)

    Clamp ke [MIN_LEVERAGE, MAX_LEVERAGE].
    """
    if sl_pct <= 0 or fraction <= 0:
        return MIN_LEVERAGE

    raw_lev = risk_per_trade / (fraction * sl_pct)
    return int(np.clip(round(raw_lev), MIN_LEVERAGE, MAX_LEVERAGE))


# ------------------------------------------------------------------
# Public: compute_position
# ------------------------------------------------------------------

def compute_position(
    df: pd.DataFrame,
    direction: str,
    winrate: float,
    risk_per_trade: float = 0.01,   # dari RISK_PER_TRADE_PCT / 100
    max_fraction: float   = 0.01,
) -> dict:
    """
    Hitung SL, TP, leverage, qty_fraction untuk satu trade.

    Args:
        df             : raw OHLCV DataFrame (ascending)
        direction      : "LONG" atau "SHORT"
        winrate        : dari backtest (winrate_up atau winrate_dn)
        risk_per_trade : max risiko per trade (default 1% = 0.01)
        max_fraction   : batas atas qty_fraction

    Returns:
        dict dengan semua parameter posisi + ringkasan MC
    """
    # Sanitize winrate
    if not (0 < winrate < 1):
        winrate = DEFAULT_WINRATE

    # 1. ATR → SL / TP / RR aktual
    sltp = compute_sltp(df, direction)
    rr   = sltp["rr_ratio"]

    # 2. Kelly dari winrate + RR aktual
    kelly_full  = _kelly_full(winrate, rr)
    is_positive = kelly_full > 0

    if is_positive:
        kelly_quarter = kelly_full * KELLY_MULTIPLIER
        kelly_capped  = max(MIN_FRACTION, min(kelly_quarter, max_fraction))
    else:
        kelly_capped = MIN_FRACTION

    # 3. Monte Carlo validasi → safe_fraction
    safe_fraction, mc = _find_safe_fraction(
        fraction=kelly_capped,
        winrate=winrate,
        rr=rr,
        max_drawdown=MAX_DRAWDOWN_PCT,
    )

    was_adjusted = safe_fraction < kelly_capped - 0.0001

    # 4. Leverage dari SL distance + fraction
    leverage = _compute_leverage(
        sl_pct=sltp["sl_pct"],
        fraction=safe_fraction,
        risk_per_trade=risk_per_trade,
    )

    edge_pct = round(kelly_full * 100, 2)

    logger.info(
        "[kelly] %s winrate=%.3f rr=%.2f → kelly=%.4f quarter=%.4f "
        "mc_dd_p5=%.3f → safe=%.4f lev=%d adjusted=%s",
        direction, winrate, rr, kelly_full, kelly_capped,
        mc["max_drawdown_p5"], safe_fraction, leverage, was_adjusted,
    )

    return {
        # Posisi
        "entry_price":    sltp["entry_price"],
        "stop_loss":      sltp["stop_loss"],
        "take_profit":    sltp["take_profit"],
        "leverage":       leverage,
        "qty_fraction":   round(safe_fraction, 6),

        # Detail kalkulasi
        "atr":            sltp["atr"],
        "sl_pct":         sltp["sl_pct"],
        "rr_ratio":       rr,
        "kelly_full":     round(kelly_full,    6),
        "kelly_quarter":  round(kelly_capped,  6),
        "edge_pct":       edge_pct,
        "is_positive_edge": is_positive,
        "was_mc_adjusted":  was_adjusted,
        "winrate":        round(winrate, 4),
        "monte_carlo":    mc,
    }


# ------------------------------------------------------------------
# Format untuk prompt AI (context only — AI tidak tentukan SL/TP/lev)
# ------------------------------------------------------------------

def format_for_prompt(pos: dict) -> str:
    mc = pos["monte_carlo"]
    edge_label = "POSITIVE ✓" if pos["is_positive_edge"] else "NEGATIVE ✗"
    adj_note   = " (MC-adjusted)" if pos.get("was_mc_adjusted") else ""
    return (
        f"  Edge              : {edge_label} ({pos['edge_pct']:+.2f}% per trade)\n"
        f"  Win Rate          : {pos['winrate']*100:.1f}%\n"
        f"  Risk/Reward       : {pos['rr_ratio']:.2f}\n"
        f"  ATR               : {pos['atr']:.6f}\n"
        f"  SL distance       : {pos['sl_pct']*100:.3f}% from entry\n"
        f"  Full Kelly        : {pos['kelly_full']*100:.2f}%\n"
        f"  Qty Fraction{adj_note}: {pos['qty_fraction']*100:.2f}%\n"
        f"  Leverage          : {pos['leverage']}x\n"
        f"  Entry             : {pos['entry_price']}\n"
        f"  Stop Loss         : {pos['stop_loss']}\n"
        f"  Take Profit       : {pos['take_profit']}\n"
        f"  Monte Carlo ({mc['n_simulations']} sims):\n"
        f"    Median outcome  : {mc['median_final']:.3f}x equity\n"
        f"    Worst 5%        : {mc['p5_final']:.3f}x equity\n"
        f"    P5 max drawdown : {mc['max_drawdown_p5']*100:.1f}%\n"
        f"    Ruin rate       : {mc['ruin_rate']*100:.1f}%"
    )
