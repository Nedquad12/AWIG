"""
board.py — Pure Python verification board, menggantikan DeepSeek di pipeline.

Melakukan 4 check secara berurutan:
  1. Winrate vs RR threshold (minimum winrate dinamis berdasarkan RR)
  2. WFV edge positif/negatif (Kelly edge)
  3. Regime profitability (apakah regime saat ini historically profitable di WFV)
  4. N signals cukup

Kalau semua pass → return action BUYING atau SELLING sesuai direction ML.
Kalau satu saja fail → return SKIP dengan alasan spesifik.

SPECIAL COIN — "2x Board":
  Koin dianggap "special" jika SEMUA 4 check memenuhi threshold 2x lipat:
    1. WR >= 2x min_wr (dari RR breakeven)
    2. Edge >= 2x MIN_EDGE_PCT  (atau >= 1.0% jika MIN_EDGE_PCT == 0)
    3. Regime: profitable_folds == ALL folds (100%, bukan hanya >= 50%)
    4. N signals >= 2x MIN_SIGNALS

  Special coin mendapat market order langsung (bukan limit order pending).
  Skor special dihitung untuk memilih koin terbaik jika ada lebih dari 1 special.

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

# Threshold 2x untuk special coin
# Edge: jika MIN_EDGE_PCT == 0, maka 2x-nya pakai floor 1.0%
SPECIAL_EDGE_FLOOR = 1.0   # floor minimum edge untuk syarat special (jika MIN_EDGE_PCT == 0)


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


# ── Special coin scoring ──────────────────────────────────────────────────────

def compute_special_score(
    pred:         dict,
    wfv_result:   dict,
    train_result: dict,
    pos_long:     dict,
    pos_short:    dict,
) -> Tuple[bool, float, dict]:
    """
    Hitung apakah koin ini "special" (memenuhi SEMUA 4 check 2x lipat threshold).

    Return:
        is_special  : bool   — True jika semua 4 check 2x terpenuhi
        score       : float  — skor komposit untuk ranking (makin tinggi makin baik)
        detail      : dict   — breakdown tiap check untuk logging/notif

    Definisi 2x:
        1. WR       >= 2 * min_wr(RR)                          (cap 1.0)
        2. edge_pct >= max(2 * MIN_EDGE_PCT, SPECIAL_EDGE_FLOOR)
        3. regime   : profitable_folds == reg_folds (100% fold profit)
        4. n_sigs   >= 2 * MIN_SIGNALS
    """
    direction = pred.get("direction", "NEUTRAL")

    if direction == "LONG":
        pos    = pos_long
        wfv_wr = wfv_result.get("after", {}).get("winrate_up", 0.0)
        n_sigs = wfv_result.get("after", {}).get("n_signal_up", 0)
    elif direction == "SHORT":
        pos    = pos_short
        wfv_wr = wfv_result.get("after", {}).get("winrate_dn", 0.0)
        n_sigs = wfv_result.get("after", {}).get("n_signal_dn", 0)
    else:
        return False, 0.0, {"reason": "NEUTRAL direction"}

    rr       = pos.get("rr_ratio", 1.0)
    edge_pct = pos.get("edge_pct", 0.0)
    is_edge  = pos.get("is_positive_edge", False)
    regime   = train_result.get("regime", "Unknown")

    reg_stats = _get_regime_stats(wfv_result, regime)
    reg_folds = reg_stats.get("folds", 0)
    reg_prof  = reg_stats.get("profitable_folds", 0)
    reg_pnl   = reg_stats.get("total_net_pnl", 0.0)

    # ── Threshold 2x ─────────────────────────────────────────────────────────
    min_wr       = _min_winrate(rr)
    min_wr_2x    = min(min_wr * 2.0, 1.0)   # cap 100%

    min_edge_2x  = max(MIN_EDGE_PCT * 2.0, SPECIAL_EDGE_FLOOR)

    min_sigs_2x  = MIN_SIGNALS * 2

    # ── Check 1: WR 2x ───────────────────────────────────────────────────────
    wr_2x_pass = wfv_wr >= min_wr_2x

    # ── Check 2: Edge 2x ─────────────────────────────────────────────────────
    edge_2x_pass = is_edge and edge_pct >= min_edge_2x

    # ── Check 3: Regime 100% fold profit ─────────────────────────────────────
    if reg_folds >= 2:
        regime_2x_pass = (reg_prof == reg_folds) and reg_pnl >= 0
    else:
        # Data kurang → anggap tidak memenuhi syarat special
        regime_2x_pass = False

    # ── Check 4: N signals 2x ────────────────────────────────────────────────
    sigs_2x_pass = n_sigs >= min_sigs_2x

    is_special = wr_2x_pass and edge_2x_pass and regime_2x_pass and sigs_2x_pass

    # ── Score komposit (untuk ranking jika ada beberapa special) ─────────────
    # Normalisasi masing-masing metrik relatif terhadap threshold 2x-nya
    wr_ratio    = wfv_wr / min_wr_2x if min_wr_2x > 0 else 0.0
    edge_ratio  = edge_pct / min_edge_2x if min_edge_2x > 0 else 0.0
    regime_ratio = (reg_prof / reg_folds) if reg_folds > 0 else 0.0
    sigs_ratio  = n_sigs / min_sigs_2x if min_sigs_2x > 0 else 0.0

    # Bobot: WR dan Edge lebih penting
    score = (wr_ratio * 0.35) + (edge_ratio * 0.35) + (regime_ratio * 0.15) + (sigs_ratio * 0.15)

    detail = {
        "is_special":     is_special,
        "score":          round(score, 4),
        "wr_2x":          {"pass": wr_2x_pass, "actual": round(wfv_wr*100, 2), "threshold": round(min_wr_2x*100, 2)},
        "edge_2x":        {"pass": edge_2x_pass, "actual": round(edge_pct, 4), "threshold": round(min_edge_2x, 4)},
        "regime_2x":      {"pass": regime_2x_pass, "actual": f"{reg_prof}/{reg_folds}", "threshold": "100% fold profit"},
        "signals_2x":     {"pass": sigs_2x_pass, "actual": n_sigs, "threshold": min_sigs_2x},
    }

    logger.info(
        "[board:special] %s is_special=%s score=%.4f | WR=%s Edge=%s Regime=%s Sigs=%s",
        pred.get("symbol", "?"), is_special, score,
        "✓" if wr_2x_pass else "✗",
        "✓" if edge_2x_pass else "✗",
        "✓" if regime_2x_pass else "✗",
        "✓" if sigs_2x_pass else "✗",
    )

    return is_special, score, detail


def format_special_verdict(symbol: str, score: float, detail: dict, pos: dict) -> str:
    """Format notifikasi Telegram untuk koin special (2x board)."""
    wr_d   = detail["wr_2x"]
    ed_d   = detail["edge_2x"]
    re_d   = detail["regime_2x"]
    si_d   = detail["signals_2x"]
    emoji  = "🟢" if pos.get("side", "BUY") == "BUY" else "🔴"

    return (
        f"⭐ <b>SPECIAL COIN — {symbol}</b> {emoji}\n"
        f"  Score     : <b>{score:.4f}</b> (2x Board)\n"
        f"  WR        : <code>{wr_d['actual']}%</code> vs threshold <code>{wr_d['threshold']}%</code> {'✓' if wr_d['pass'] else '✗'}\n"
        f"  Edge      : <code>{ed_d['actual']:+.2f}%</code> vs threshold <code>{ed_d['threshold']:.2f}%</code> {'✓' if ed_d['pass'] else '✗'}\n"
        f"  Regime    : <code>{re_d['actual']}</code> fold profit (100% req) {'✓' if re_d['pass'] else '✗'}\n"
        f"  Signals   : <code>{si_d['actual']}</code> vs min <code>{si_d['threshold']}</code> {'✓' if si_d['pass'] else '✗'}\n"
        f"  → 🚀 <b>MARKET ORDER LANGSUNG</b>"
    )


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
