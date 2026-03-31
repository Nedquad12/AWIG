"""
ml/kelly.py — Kelly Criterion + Monte Carlo position sizing.

FIXES:
  1. Tambah hard floor untuk winrate: tidak boleh pakai winrate dari
     backtest yang suspiciously tinggi (> 75%) — kemungkinan overfitting.
  2. Tambah sanity check: jika n_signal terlalu sedikit (< MIN_SIGNAL_SAMPLE),
     winrate tidak reliable → fallback ke DEFAULT_WINRATE yang konservatif.
  3. Kelly multiplier dikurangi dari 0.25 ke 0.20 (lebih konservatif)
     karena winrate kita berasal dari backtest, bukan live trading.
  4. Dokumentasi lebih jelas tentang asumsi dan limitasi.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kelly
KELLY_MULTIPLIER  = 0.20   # FIX: diturunkan dari 0.25 ke 0.20 (lebih konservatif)
                            # Alasan: winrate dari backtest selalu lebih optimis dari live
MIN_FRACTION      = 0.005  # 0.5%

# Winrate guard
MIN_SIGNAL_SAMPLE = 20     # butuh minimal 20 signal untuk winrate yang reliable
MAX_WINRATE       = 0.72   # cap winrate: lebih dari ini → suspiciously high → overfitting
DEFAULT_WINRATE   = 0.48   # konservatif: sedikit di bawah 50% agar Kelly tidak agresif

# Monte Carlo
MAX_DRAWDOWN_PCT  = 0.15   # FIX: diturunkan dari 20% ke 15% (lebih konservatif untuk live)
MC_SIMULATIONS    = 1000
MC_TRADES         = 100

# ATR
ATR_PERIOD        = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0

# Leverage
MIN_LEVERAGE      = 3
MAX_LEVERAGE      = 10     # FIX: diturunkan dari 15 ke 10 untuk crypto volatility

# Default RR
DEFAULT_RR        = 2.0


# ------------------------------------------------------------------
# Validasi dan sanitize winrate dari backtest
# ------------------------------------------------------------------

def sanitize_winrate(winrate: float, n_signals: int, label: str = "") -> tuple[float, str]:
    """
    Validasi dan sanitize winrate dari backtest sebelum dipakai Kelly.

    Returns:
        (sanitized_winrate, warning_message)
        warning_message = "" jika tidak ada masalah
    """
    warning = ""

    # Tidak valid
    if not (0 < winrate < 1):
        warning = f"Winrate {winrate:.3f} tidak valid → fallback {DEFAULT_WINRATE}"
        logger.warning("[kelly] %s %s", label, warning)
        return DEFAULT_WINRATE, warning

    # Sample terlalu sedikit → winrate tidak reliable
    if n_signals < MIN_SIGNAL_SAMPLE:
        warning = (
            f"Sample terlalu sedikit ({n_signals} < {MIN_SIGNAL_SAMPLE}) "
            f"→ winrate tidak reliable → fallback {DEFAULT_WINRATE}"
        )
        logger.warning("[kelly] %s %s", label, warning)
        return DEFAULT_WINRATE, warning

    # Suspiciously high → kemungkinan overfitting
    if winrate > MAX_WINRATE:
        capped = MAX_WINRATE
        warning = (
            f"Winrate {winrate:.3f} > cap {MAX_WINRATE} → "
            f"kemungkinan overfitting → cap ke {capped}"
        )
        logger.warning("[kelly] %s %s", label, warning)
        return capped, warning

    return winrate, warning


# ------------------------------------------------------------------
# ATR calculation
# ------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    if len(df) < period + 1:
        return float(np.mean(df["high"].tail(period).values - df["low"].tail(period).values))

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    tr = []
    for i in range(1, len(df)):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i]  - closes[i - 1])
        tr.append(max(hl, hpc, lpc))

    return float(np.mean(tr[-period:]))


# ------------------------------------------------------------------
# SL / TP dari ATR
# ------------------------------------------------------------------

def compute_sltp(
    df: pd.DataFrame,
    direction: str,
    sl_multiplier: float = ATR_SL_MULTIPLIER,
    tp_multiplier: float = ATR_TP_MULTIPLIER,
) -> dict:
    entry = float(df["close"].iloc[-1])
    atr   = _compute_atr(df)

    sl_dist = atr * sl_multiplier
    tp_dist = atr * tp_multiplier
    rr      = tp_dist / sl_dist

    if direction == "LONG":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    sl_pct = sl_dist / entry

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
        "median_final":     round(float(np.median(final_equities)),        4),
        "p5_final":         round(float(np.percentile(final_equities, 5)), 4),
        "p95_final":        round(float(np.percentile(final_equities, 95)),4),
        "max_drawdown_p5":  round(float(np.percentile(max_drawdowns, 95)), 4),
        "max_drawdown_med": round(float(np.median(max_drawdowns)),         4),
        "ruin_rate":        round(float(np.mean(final_equities < 0.5)),    4),
        "n_simulations":    n_simulations,
        "n_trades":         n_trades,
    }


def _find_safe_fraction(
    fraction: float,
    winrate: float,
    rr: float,
    max_drawdown: float = MAX_DRAWDOWN_PCT,
    step: float = 0.001,
) -> tuple[float, dict]:
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
# Leverage
# ------------------------------------------------------------------

def _compute_leverage(sl_pct: float, fraction: float, risk_per_trade: float) -> int:
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
    n_signals: int = 0,             # BARU: jumlah sinyal di backtest untuk validasi
    risk_per_trade: float = 0.01,
    max_fraction: float   = 0.01,
) -> dict:
    """
    Hitung SL, TP, leverage, qty_fraction untuk satu trade.

    PERUBAHAN:
      - Winrate disanitize sebelum dipakai Kelly
      - Kelly multiplier lebih konservatif (0.20)
      - Max leverage diturunkan ke 10
      - Max drawdown target diturunkan ke 15%
    """
    # Sanitize winrate — ini fix utama
    safe_wr, wr_warning = sanitize_winrate(winrate, n_signals, label=f"{direction} {df['close'].iloc[-1]:.2f}")

    # ATR → SL / TP / RR
    sltp = compute_sltp(df, direction)
    rr   = sltp["rr_ratio"]

    # Kelly
    kelly_full    = _kelly_full(safe_wr, rr)
    is_positive   = kelly_full > 0

    if is_positive:
        kelly_quarter = kelly_full * KELLY_MULTIPLIER
        kelly_capped  = max(MIN_FRACTION, min(kelly_quarter, max_fraction))
    else:
        kelly_capped = MIN_FRACTION

    # Monte Carlo validasi
    safe_fraction, mc = _find_safe_fraction(
        fraction=kelly_capped,
        winrate=safe_wr,
        rr=rr,
        max_drawdown=MAX_DRAWDOWN_PCT,
    )

    was_adjusted = safe_fraction < kelly_capped - 0.0001

    # Leverage
    leverage = _compute_leverage(
        sl_pct=sltp["sl_pct"],
        fraction=safe_fraction,
        risk_per_trade=risk_per_trade,
    )

    edge_pct = round(kelly_full * 100, 2)

    logger.info(
        "[kelly] %s winrate=%.3f (sanitized from %.3f, n=%d) rr=%.2f → "
        "kelly=%.4f quarter=%.4f mc_dd_p5=%.3f → safe=%.4f lev=%d%s",
        direction, safe_wr, winrate, n_signals, rr, kelly_full, kelly_capped,
        mc["max_drawdown_p5"], safe_fraction, leverage,
        f" | WARNING: {wr_warning}" if wr_warning else "",
    )

    return {
        "entry_price":       sltp["entry_price"],
        "stop_loss":         sltp["stop_loss"],
        "take_profit":       sltp["take_profit"],
        "leverage":          leverage,
        "qty_fraction":      round(safe_fraction, 6),

        "atr":               sltp["atr"],
        "sl_pct":            sltp["sl_pct"],
        "rr_ratio":          rr,
        "kelly_full":        round(kelly_full,    6),
        "kelly_quarter":     round(kelly_capped,  6),
        "edge_pct":          edge_pct,
        "is_positive_edge":  is_positive,
        "was_mc_adjusted":   was_adjusted,
        "winrate":           round(safe_wr, 4),
        "winrate_raw":       round(winrate, 4),    # BARU: winrate asli sebelum sanitize
        "winrate_warning":   wr_warning,           # BARU: peringatan jika ada adjustment
        "n_signals":         n_signals,
        "monte_carlo":       mc,
    }


# ------------------------------------------------------------------
# Format untuk prompt AI
# ------------------------------------------------------------------

def format_for_prompt(pos: dict) -> str:
    mc          = pos["monte_carlo"]
    edge_label  = "POSITIVE ✓" if pos["is_positive_edge"] else "NEGATIVE ✗"
    adj_note    = " (MC-adjusted)" if pos.get("was_mc_adjusted") else ""
    wr_note     = f" ⚠️ RAW={pos['winrate_raw']*100:.1f}%" if pos.get("winrate_warning") else ""
    n_sig_note  = f" (n={pos['n_signals']} signals)" if pos.get("n_signals", 0) > 0 else ""

    return (
        f"  Edge              : {edge_label} ({pos['edge_pct']:+.2f}% per trade)\n"
        f"  Win Rate          : {pos['winrate']*100:.1f}%{wr_note}{n_sig_note}\n"
        f"  Risk/Reward       : {pos['rr_ratio']:.2f}\n"
        f"  ATR               : {pos['atr']:.6f}\n"
        f"  SL distance       : {pos['sl_pct']*100:.3f}% from entry\n"
        f"  Full Kelly        : {pos['kelly_full']*100:.2f}%\n"
        f"  Qty Fraction{adj_note}: {pos['qty_fraction']*100:.2f}%\n"
        f"  Leverage          : {pos['leverage']}x\n"
        f"  Entry             : {pos['entry_price']}\n"
        f"  Stop Loss         : {pos['stop_loss']}\n"
        f"  Take Profit       : {pos['take_profit']}\n"
        f"  Monte Carlo ({mc['n_simulations']} sims, max_dd_target={MAX_DRAWDOWN_PCT*100:.0f}%):\n"
        f"    Median outcome  : {mc['median_final']:.3f}x equity\n"
        f"    Worst 5%        : {mc['p5_final']:.3f}x equity\n"
        f"    P5 max drawdown : {mc['max_drawdown_p5']*100:.1f}%\n"
        f"    Ruin rate       : {mc['ruin_rate']*100:.1f}%"
    )
