import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KELLY_MULTIPLIER  = 0.20  
MIN_FRACTION      = 0.005


MIN_SIGNAL_SAMPLE = 20   
MAX_WINRATE       = 0.72 
DEFAULT_WINRATE   = 0.48  

MAX_DRAWDOWN_PCT  = 0.15
MC_SIMULATIONS    = 1000
MC_TRADES         = 100

ATR_PERIOD        = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0 

MIN_LEVERAGE      = 15
MAX_LEVERAGE      = 30     

RISK_PER_TRADE_PCT = 0.01  


def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD, train_end: int = 850) -> float:
    train_df = df.iloc[:train_end] if len(df) >= train_end else df

    if len(train_df) < period + 1:
        return float(np.mean(
            train_df["high"].tail(period).values - train_df["low"].tail(period).values
        ))

    highs  = train_df["high"].values
    lows   = train_df["low"].values
    closes = train_df["close"].values

    tr = []
    for i in range(1, len(train_df)):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i]  - closes[i - 1])
        tr.append(max(hl, hpc, lpc))

    return float(np.mean(tr[-period:]))

def sanitize_winrate(winrate: float, n_signals: int, label: str = "") -> tuple[float, str]:
    warning = ""

    if not (0 < winrate < 1):
        warning = f"Winrate {winrate:.3f} tidak valid → fallback {DEFAULT_WINRATE}"
        logger.warning("[kelly] %s %s", label, warning)
        return DEFAULT_WINRATE, warning

    if n_signals < MIN_SIGNAL_SAMPLE:
        warning = (
            f"Sample terlalu sedikit ({n_signals} < {MIN_SIGNAL_SAMPLE}) "
            f"→ winrate tidak reliable → fallback {DEFAULT_WINRATE}"
        )
        logger.warning("[kelly] %s %s", label, warning)
        return DEFAULT_WINRATE, warning

    if winrate > MAX_WINRATE:
        warning = (
            f"Winrate {winrate:.3f} > cap {MAX_WINRATE} "
            f"→ kemungkinan test set terlalu kecil → cap ke {MAX_WINRATE}"
        )
        logger.warning("[kelly] %s %s", label, warning)
        return MAX_WINRATE, warning

    return winrate, warning

def compute_sltp(
    df: pd.DataFrame,
    direction: str,
    sl_multiplier: float = ATR_SL_MULTIPLIER,
    tp_multiplier: float = ATR_TP_MULTIPLIER,
    train_end: int = 850,
) -> dict:
    entry = float(df["close"].iloc[-1])
    atr   = _compute_atr(df, train_end=train_end)

    sl_dist = atr * sl_multiplier
    tp_dist = atr * tp_multiplier
    rr      = tp_dist / sl_dist

    if direction == "LONG":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    sl_pct = sl_dist / entry  # berapa persen dari entry ke SL

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

def _kelly_full(winrate: float, rr: float) -> float:
    p = winrate
    q = 1.0 - p
    b = rr
    return (p * b - q) / b

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
        "median_final":     round(float(np.median(final_equities)),         4),
        "p5_final":         round(float(np.percentile(final_equities, 5)),  4),
        "p95_final":        round(float(np.percentile(final_equities, 95)), 4),
        "max_drawdown_p5":  round(float(np.percentile(max_drawdowns, 95)),  4),
        "max_drawdown_med": round(float(np.median(max_drawdowns)),          4),
        "ruin_rate":        round(float(np.mean(final_equities < 0.5)),     4),
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

def _compute_leverage(sl_pct: float) -> int:
    if sl_pct <= 0:
        return MIN_LEVERAGE

    raw_lev = RISK_PER_TRADE_PCT / sl_pct
    return int(np.clip(round(raw_lev), MIN_LEVERAGE, MAX_LEVERAGE))

def compute_position(
    df: pd.DataFrame,
    direction: str,
    winrate: float,
    n_signals: int = 0,
    risk_per_trade: float = RISK_PER_TRADE_PCT,
    max_fraction: float   = RISK_PER_TRADE_PCT,
    train_end: int = 850,
    kelly_multiplier_override: float | None = None,
) -> dict:
    safe_wr, wr_warning = sanitize_winrate(
        winrate, n_signals,
        label=f"{direction} @{df['close'].iloc[-1]:.4f}"
    )
    sltp = compute_sltp(df, direction, train_end=train_end)
    rr   = sltp["rr_ratio"]

    kelly_mult    = kelly_multiplier_override if kelly_multiplier_override is not None else KELLY_MULTIPLIER
    kelly_full    = _kelly_full(safe_wr, rr)
    is_positive   = kelly_full > 0

    if is_positive:
        kelly_quarter = kelly_full * kelly_mult
        kelly_capped  = max(MIN_FRACTION, min(kelly_quarter, max_fraction))
    else:
        kelly_capped = MIN_FRACTION
        
    safe_fraction, mc = _find_safe_fraction(
        fraction=kelly_capped,
        winrate=safe_wr,
        rr=rr,
        max_drawdown=MAX_DRAWDOWN_PCT,
    )

    was_adjusted = safe_fraction < kelly_capped - 0.0001

    leverage = _compute_leverage(sl_pct=sltp["sl_pct"])

    edge_pct = round(kelly_full * 100, 2)

    logger.info(
        "[kelly] %s wr=%.3f (raw=%.3f, n=%d) rr=%.2f kelly=%.4f "
        "quarter=%.4f mc_dd_p5=%.3f → frac=%.4f lev=%dx%s",
        direction, safe_wr, winrate, n_signals, rr,
        kelly_full, kelly_capped, mc["max_drawdown_p5"],
        safe_fraction, leverage,
        f" | {wr_warning}" if wr_warning else "",
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
        "kelly_full":        round(kelly_full,   6),
        "kelly_quarter":     round(kelly_capped, 6),
        "edge_pct":          edge_pct,
        "is_positive_edge":  is_positive,
        "was_mc_adjusted":   was_adjusted,
        "winrate":           round(safe_wr, 4),
        "winrate_raw":       round(winrate, 4),
        "winrate_warning":   wr_warning,
        "n_signals":         n_signals,
        "monte_carlo":       mc,
        "risk_per_trade_pct": risk_per_trade * 100,
    }

def format_for_prompt(pos: dict) -> str:
    mc         = pos["monte_carlo"]
    edge_label = "POSITIVE ✓" if pos["is_positive_edge"] else "NEGATIVE ✗"
    adj_note   = " (MC-adjusted)" if pos.get("was_mc_adjusted") else ""
    wr_note    = f" ⚠️ RAW={pos['winrate_raw']*100:.1f}%" if pos.get("winrate_warning") else ""
    n_sig_note = f" (n={pos['n_signals']} signals)" if pos.get("n_signals", 0) > 0 else ""

    return (
        f"  Edge              : {edge_label} ({pos['edge_pct']:+.2f}% per trade)\n"
        f"  Win Rate          : {pos['winrate']*100:.1f}%{wr_note}{n_sig_note}\n"
        f"  Risk/Reward       : {pos['rr_ratio']:.2f}\n"
        f"  ATR (train data)  : {pos['atr']:.6f}\n"
        f"  SL distance       : {pos['sl_pct']*100:.3f}% from entry\n"
        f"  Full Kelly        : {pos['kelly_full']*100:.2f}%\n"
        f"  Qty Fraction{adj_note}: {pos['qty_fraction']*100:.2f}%\n"
        f"  Leverage          : {pos['leverage']}x  (margin risk = {pos.get('risk_per_trade_pct', 1.0):.1f}% modal)\n"
        f"  Entry             : {pos['entry_price']}\n"
        f"  Stop Loss         : {pos['stop_loss']}\n"
        f"  Take Profit       : {pos['take_profit']}\n"
        f"  Monte Carlo ({mc['n_simulations']} sims, max_dd_target={MAX_DRAWDOWN_PCT*100:.0f}%):\n"
        f"    Median outcome  : {mc['median_final']:.3f}x equity\n"
        f"    Worst 5%        : {mc['p5_final']:.3f}x equity\n"
        f"    P5 max drawdown : {mc['max_drawdown_p5']*100:.1f}%\n"
        f"    Ruin rate       : {mc['ruin_rate']*100:.1f}%"
    )
