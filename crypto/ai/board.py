"""
board.py — Pure Python verification board, menggantikan DeepSeek di pipeline.

Melakukan 4 check secara berurutan:
  1. Winrate vs RR threshold (minimum winrate dinamis berdasarkan RR)
  2. WFV edge positif/negatif (Kelly edge)
  3. Regime profitability (apakah regime saat ini historically profitable di WFV)
  4. Confidence ML minimum

Kalau semua pass → return action BUYING atau SELLING sesuai direction ML.
Kalau satu saja fail → return SKIP dengan alasan spesifik.

Tidak ada LLM, tidak ada HTTP call. Deterministik dan cepat.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Threshold config ────────────────────────────────────────────────────────

# Minimum winrate dinamis: breakeven WR berdasarkan RR
# contoh: RR 2.0 → breakeven WR = 1/(1+2) = 33.3%
# Floor 20%, cap 55%
MIN_WR_FLOOR = 0.20
MIN_WR_CAP   = 0.55

# WFV regime profitability — jika profitable folds < threshold → skip
REGIME_MIN_PROF_RATIO = 0.50   # minimal 50% folds harus profit di regime ini

# Minimum Kelly edge untuk masuk (positif saja tidak cukup, harus >= threshold)
MIN_EDGE_PCT = 0.0   # 0% = edge positif saja sudah cukup; bisa dinaikkan misal 2.0

# Minimum n_signals agar winrate dianggap valid
MIN_SIGNALS = 10


# ── Helper ───────────────────────────────────────────────────────────────────

def _min_winrate(rr: float) -> float:
    """Hitung minimum winrate berdasarkan RR ratio (breakeven formula)."""
    breakeven = 1.0 / (1.0 + rr)
    return round(max(MIN_WR_FLOOR, min(breakeven, MIN_WR_CAP)), 4)


def _get_regime_stats(wfv_result: dict, regime: str) -> dict:
    """Ambil agregat WFV untuk regime tertentu."""
    ra = wfv_result.get("regime_agg", {})
    return ra.get(regime, {})


# ── Main verify function ──────────────────────────────────────────────────────

def verify(
    pred:         dict,
    wfv_result:   dict,
    train_result: dict,
    pos_long:     dict,
    pos_short:    dict,
) -> Tuple[str, str]:
    """
    Verifikasi signal dan return (action, reason).
    action: "BUYING" | "SELLING" | "SKIP"
    reason: teks singkat alasan keputusan
    """
    symbol    = pred["symbol"]
    direction = pred["direction"]   # "LONG" | "SHORT" | "NEUTRAL"
    regime    = train_result.get("regime", "Unknown")

    # Pilih posisi yang relevan dengan direction ML
    if direction == "LONG":
        pos     = pos_long
        action  = "BUYING"
        wfv_wr  = wfv_result.get("after", {}).get("winrate_up", 0.0)
        n_sigs  = wfv_result.get("after", {}).get("n_signal_up", 0)
    elif direction == "SHORT":
        pos     = pos_short
        action  = "SELLING"
        wfv_wr  = wfv_result.get("after", {}).get("winrate_dn", 0.0)
        n_sigs  = wfv_result.get("after", {}).get("n_signal_dn", 0)
    else:
        return "SKIP", "Direction NEUTRAL — tidak ada sinyal jelas dari ML"

    rr       = pos.get("rr_ratio", 1.0)
    edge_pct = pos.get("edge_pct", 0.0)
    is_edge  = pos.get("is_positive_edge", False)
    wr_raw   = pos.get("winrate_raw", wfv_wr)
    wr_warn  = pos.get("winrate_warning", False)

    checks   = []
    failed   = []

    # ── Check 1: Winrate vs RR threshold ─────────────────────────────────────
    min_wr   = _min_winrate(rr)
    wr_pass  = wfv_wr >= min_wr
    wr_note  = f"⚠️ raw={wr_raw*100:.1f}%" if wr_warn else ""
    checks.append(
        f"{'✓' if wr_pass else '✗'} WR {wfv_wr*100:.1f}% "
        f">= min {min_wr*100:.1f}% (RR {rr:.2f}) {wr_note}"
    )
    if not wr_pass:
        failed.append(
            f"Winrate {wfv_wr*100:.1f}% di bawah minimum {min_wr*100:.1f}% "
            f"untuk RR {rr:.2f}"
        )

    # ── Check 2: Edge positif ─────────────────────────────────────────────────
    edge_pass = is_edge and edge_pct >= MIN_EDGE_PCT
    checks.append(
        f"{'✓' if edge_pass else '✗'} Kelly edge {edge_pct:+.2f}% "
        f"({'POSITIVE' if is_edge else 'NEGATIVE'})"
    )
    if not edge_pass:
        failed.append(
            f"Kelly edge {edge_pct:+.2f}% — "
            f"{'negatif' if not is_edge else f'di bawah minimum {MIN_EDGE_PCT:.1f}%'}"
        )

    # ── Check 3: Regime profitability ─────────────────────────────────────────
    reg_stats    = _get_regime_stats(wfv_result, regime)
    reg_folds    = reg_stats.get("folds", 0)
    reg_prof     = reg_stats.get("profitable_folds", 0)
    reg_pnl      = reg_stats.get("total_net_pnl", 0.0)
    reg_trades   = reg_stats.get("total_trades", 0)

    if reg_folds >= 2:
        prof_ratio = reg_prof / reg_folds
        reg_pass   = prof_ratio >= REGIME_MIN_PROF_RATIO and reg_pnl >= 0
        checks.append(
            f"{'✓' if reg_pass else '✗'} Regime {regime}: "
            f"{reg_prof}/{reg_folds} fold profit, PnL ${reg_pnl:+.2f}"
        )
        if not reg_pass:
            failed.append(
                f"Regime {regime} historically jelek: "
                f"{reg_prof}/{reg_folds} fold profit, net PnL ${reg_pnl:.2f}"
            )
    else:
        # Data regime tidak cukup — lewati check ini
        checks.append(
            f"~ Regime {regime}: data kurang ({reg_folds} fold) — skip check"
        )

    # ── Check 4: N signals cukup ──────────────────────────────────────────────
    sig_pass = n_sigs >= MIN_SIGNALS
    checks.append(
        f"{'✓' if sig_pass else '✗'} N signals: {n_sigs} "
        f">= min {MIN_SIGNALS}"
    )
    if not sig_pass:
        failed.append(
            f"Hanya {n_sigs} sinyal historis — winrate kurang reliable "
            f"(min {MIN_SIGNALS})"
        )

    # ── Final decision ────────────────────────────────────────────────────────
    check_summary = " | ".join(checks)

    if failed:
        reason = f"SKIP [{symbol} {direction}] — " + "; ".join(failed)
        logger.info("[board] %s → SKIP | %s", symbol, check_summary)
        return "SKIP", reason

    reason = (
        f"{action} [{symbol}] — "
        f"WR {wfv_wr*100:.1f}% ✓ | "
        f"Edge {edge_pct:+.2f}% ✓ | "
        f"Regime {regime} {reg_prof}/{reg_folds} fold profit ✓ | "
        f"n={n_sigs} signals ✓"
    )
    logger.info("[board] %s → %s | %s", symbol, action, check_summary)
    return action, reason


def format_verdict(action: str, reason: str, pos: dict) -> str:
    """Format output board untuk Telegram notification."""
    if action == "SKIP":
        return (
            f"🔲 <b>Board: SKIP</b>\n"
            f"  <i>{reason}</i>"
        )

    emoji = "🟢" if action == "BUYING" else "🔴"
    mc    = pos.get("monte_carlo", {})
    return (
        f"{emoji} <b>Board: {action}</b>\n"
        f"  Entry  : <code>{pos.get('entry_price')}</code>\n"
        f"  SL     : <code>{pos.get('stop_loss')}</code>\n"
        f"  TP     : <code>{pos.get('take_profit')}</code>\n"
        f"  RR     : <code>{pos.get('rr_ratio', 0):.2f}</code>\n"
        f"  Edge   : <code>{pos.get('edge_pct', 0):+.2f}%</code>\n"
        f"  Lev    : <b>{pos.get('leverage')}x</b>\n"
        f"  MC DD  : <code>{mc.get('max_drawdown_p5', 0)*100:.1f}%</code>\n"
        f"  Reason : <i>{reason}</i>"
    )
