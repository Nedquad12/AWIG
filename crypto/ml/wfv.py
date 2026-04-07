"""
ml/wfv.py — Walk-Forward Validation (menggantikan backtest.py).

Flow per fold:
  1. Ambil window train (500 candle) dari feat_df
  2. Deteksi regime dari training window
  3. Ambil parameter model sesuai regime
  4. Evaluasi bobot (weights_after) di test window (100 candle)
  5. Simulasikan setiap signal sebagai trade, hitung PnL net (fee + slippage)
  6. Geser window sebesar STEP (100 candle), ulangi

Output:
  - List fold results (regime, WR, PnL, n_trades)
  - Agregat per regime
  - Summary text untuk AI
"""

import logging

import numpy as np
import pandas as pd

from ml.weight_manager  import apply_weights
from ml.regime_detector import detect_regime, REGIME_PARAMS, TRENDING, SIDEWAYS, VOLATILE
from ml.cost_model      import compute_trade_pnl, FEE_RATE_RT, FALLBACK_SLIPPAGE

logger = logging.getLogger(__name__)

# ── WFV config ────────────────────────────────────────────────────────
TRAIN_WINDOW  = 500    # candle per training window
TEST_WINDOW   = 100    # candle per test window (OOS)
STEP          = 100    # geser sebesar ini setiap fold

# ── Backtest split (untuk backward compat dengan analyst.py) ─────────
TRAIN_END_IDX = 850    # tetap dipertahankan untuk split trainer vs OOS

# ── Simulasi trade config ─────────────────────────────────────────────
SIGNAL_UP     =  1.0
SIGNAL_DOWN   = -1.0
SIM_MARGIN    = 100.0          # modal simulasi $100 per trade (bukan per sesi)
SIM_LEVERAGE  = 10             # leverage default untuk simulasi
FEATURES_CORE = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc"]


# ------------------------------------------------------------------
# Evaluasi satu test window: signal → trades → PnL
# ------------------------------------------------------------------

def _compute_atr_at(raw_df: pd.DataFrame, candle_idx: int, period: int = 14) -> float:
    """
    Hitung ATR aktual dari raw_df pada posisi candle_idx.
    Dipakai untuk menentukan SL/TP yang realistis saat simulasi trade.
    """
    end   = candle_idx + 1
    start = max(0, end - period - 1)
    sub   = raw_df.iloc[start:end]
    if len(sub) < 2:
        # Fallback: pakai H-L range
        row = raw_df.iloc[candle_idx]
        return float(row["high"] - row["low"])

    highs  = sub["high"].values
    lows   = sub["low"].values
    closes = sub["close"].values
    tr = []
    for i in range(1, len(sub)):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i]  - closes[i - 1])
        lpc = abs(lows[i]   - closes[i - 1])
        tr.append(max(hl, hpc, lpc))
    return float(np.mean(tr)) if tr else float(highs[-1] - lows[-1])


def _simulate_trades(
    test_df: pd.DataFrame,
    weights: dict,
    symbol: str,
    regime: str,
    raw_df: pd.DataFrame,
    test_raw_start: int,
    fetch_slippage: bool = True,
) -> list[dict]:
    """
    Simulasikan trades dari test window.
      - ATR dihitung dari raw_df aktual (bukan proxy 0.5%)
      - SL = 1.5 × ATR, TP = 3.0 × ATR (konsisten dengan kelly.py)
      - Exit: TP jika label sesuai arah, SL jika tidak

    test_raw_start: index di raw_df yang berkorespondensi dengan baris pertama test_df
                    Dipakai untuk lookup ATR per baris.
    """
    trades = []

    for local_i, (_, row) in enumerate(test_df.iterrows()):
        try:
            total = apply_weights({f: row[f] for f in FEATURES_CORE if f in row}, weights)
        except Exception:
            continue

        if total >= SIGNAL_UP:
            direction = "LONG"
        elif total <= SIGNAL_DOWN:
            direction = "SHORT"
        else:
            continue

        label = int(row.get("label", 0))
        price = float(row.get("price", 0))
        if price <= 0:
            continue

        # ATR aktual dari raw_df pada posisi ini
        raw_idx = test_raw_start + local_i
        atr_est = _compute_atr_at(raw_df, min(raw_idx, len(raw_df) - 1))

        # SL/TP konsisten dengan kelly.py (1.5× dan 3.0× ATR)
        sl_dist = atr_est * 1.5
        tp_dist = atr_est * 3.0

        if direction == "LONG":
            tp_price   = price + tp_dist
            sl_price   = price - sl_dist
            exit_price = tp_price if label == 1 else sl_price
        else:
            tp_price   = price - tp_dist
            sl_price   = price + sl_dist
            exit_price = tp_price if label == -1 else sl_price

        pnl_result = compute_trade_pnl(
            symbol        = symbol,
            direction     = direction,
            entry_price   = price,
            exit_price    = exit_price,
            margin_usdt   = SIM_MARGIN,
            leverage      = SIM_LEVERAGE,
            fetch_slippage= fetch_slippage,
        )

        trades.append({
            "direction":    direction,
            "entry_price":  price,
            "exit_price":   exit_price,
            "label":        label,
            "atr":          round(atr_est, 8),
            "win":          (direction == "LONG" and label == 1) or
                            (direction == "SHORT" and label == -1),
            **pnl_result,
        })

    return trades


# ------------------------------------------------------------------
# Satu fold WFV
# ------------------------------------------------------------------

def _run_fold(
    feat_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    weights_before: dict,
    weights_after: dict,
    symbol: str,
    fold_idx: int,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    fetch_slippage: bool = True,
) -> dict:
    train_window_df = raw_df.iloc[train_start:train_end]
    test_feat_df    = feat_df.iloc[test_start:test_end].reset_index(drop=True)

    if len(train_window_df) < 50 or len(test_feat_df) < 5:
        return {"ok": False, "fold": fold_idx, "reason": "window terlalu kecil"}

    # Deteksi regime dari training window
    regime_info = detect_regime(train_window_df)
    regime      = regime_info["regime"]

    # Simulasikan trades dengan weights_after
    # test_start di feat_df berkorespondensi dengan raw_df di index yang sama
    trades = _simulate_trades(
        test_feat_df, weights_after, symbol, regime,
        raw_df=raw_df,
        test_raw_start=test_start,
        fetch_slippage=fetch_slippage,
    )

    n_trades = len(trades)
    n_wins   = sum(1 for t in trades if t["win"])
    winrate  = n_wins / n_trades if n_trades > 0 else 0.0
    net_pnl  = sum(t["net_pnl"] for t in trades)
    gross_pnl= sum(t["gross_pnl"] for t in trades)
    total_fee= sum(t["fee_usdt"] for t in trades)
    total_slip=sum(t["slippage_usdt"] for t in trades)
    avg_slip = sum(t["slippage_pct"] for t in trades) / n_trades if n_trades > 0 else 0

    # Signal count dari feat_df
    totals = []
    for _, row in test_feat_df.iterrows():
        try:
            totals.append(apply_weights({f: row[f] for f in FEATURES_CORE if f in row}, weights_after))
        except Exception:
            totals.append(0.0)

    n_signal_up = sum(1 for t in totals if t >= SIGNAL_UP)
    n_signal_dn = sum(1 for t in totals if t <= SIGNAL_DOWN)

    logger.info(
        "[wfv] Fold %d | %s | regime=%s | WR=%.1f%% | PnL_net=$%.2f | trades=%d",
        fold_idx, symbol, regime, winrate * 100, net_pnl, n_trades,
    )

    return {
        "ok":           True,
        "fold":         fold_idx,
        "symbol":       symbol,
        "regime":       regime,
        "regime_info":  regime_info,
        "train_range":  (train_start, train_end),
        "test_range":   (test_start,  test_end),
        "n_trades":     n_trades,
        "n_wins":       n_wins,
        "winrate":      round(winrate,   4),
        "net_pnl":      round(net_pnl,   4),
        "gross_pnl":    round(gross_pnl, 4),
        "total_fee":    round(total_fee, 4),
        "total_slip":   round(total_slip,4),
        "avg_slip_pct": round(avg_slip,  6),
        "n_signal_up":  n_signal_up,
        "n_signal_dn":  n_signal_dn,
        "trades":       trades,
    }


# ------------------------------------------------------------------
# Public: run_wfv
# ------------------------------------------------------------------

def run_wfv(
    train_result: dict,
    fetch_slippage: bool = True,
) -> dict:
    """
    Jalankan Walk-Forward Validation.

    Walk-forward config:
      Train window : TRAIN_WINDOW (500 candle)
      Test window  : TEST_WINDOW  (100 candle)
      Step         : STEP         (100 candle)

    Fold i:
      train: [i*STEP, i*STEP + TRAIN_WINDOW)
      test : [i*STEP + TRAIN_WINDOW, i*STEP + TRAIN_WINDOW + TEST_WINDOW)

    Fold berhenti saat test_end > len(feat_df).
    """
    feat_df        = train_result["feature_df"]
    raw_df         = train_result["raw_df"]
    weights_before = train_result["weights_before"]
    weights_after  = train_result["weights_after"]
    symbol         = train_result["symbol"]
    interval       = train_result["interval"]
    n_total        = len(feat_df)

    logger.info(
        "[wfv] %s — start WFV (train=%d, test=%d, step=%d, total_rows=%d)",
        symbol, TRAIN_WINDOW, TEST_WINDOW, STEP, n_total,
    )

    folds = []
    fold_idx = 0
    start = 0

    while True:
        train_start = start
        train_end   = start + TRAIN_WINDOW
        test_start  = train_end
        test_end    = test_start + TEST_WINDOW

        if test_end > n_total:
            break

        fold = _run_fold(
            feat_df        = feat_df,
            raw_df         = raw_df,
            weights_before = weights_before,
            weights_after  = weights_after,
            symbol         = symbol,
            fold_idx       = fold_idx,
            train_start    = train_start,
            train_end      = train_end,
            test_start     = test_start,
            test_end       = test_end,
            fetch_slippage = fetch_slippage,
        )
        folds.append(fold)
        fold_idx += 1
        start    += STEP

    # Agregat keseluruhan
    ok_folds    = [f for f in folds if f.get("ok")]
    n_folds     = len(ok_folds)
    total_trades= sum(f["n_trades"]  for f in ok_folds)
    total_wins  = sum(f["n_wins"]    for f in ok_folds)
    total_pnl   = sum(f["net_pnl"]   for f in ok_folds)
    overall_wr  = total_wins / total_trades if total_trades > 0 else 0.0

    # Agregat per regime
    from collections import defaultdict
    regime_agg: dict[str, dict] = defaultdict(lambda: {
        "folds": 0, "profitable_folds": 0,
        "total_net_pnl": 0.0, "total_trades": 0, "total_wins": 0,
        "winrate": 0.0,
    })
    for f in ok_folds:
        reg = f["regime"]
        a   = regime_agg[reg]
        a["folds"]           += 1
        a["total_net_pnl"]   += f["net_pnl"]
        a["total_trades"]    += f["n_trades"]
        a["total_wins"]      += f["n_wins"]
        if f["net_pnl"] > 0:
            a["profitable_folds"] += 1

    for reg, a in regime_agg.items():
        nt = a["total_trades"]
        a["winrate"] = round(a["total_wins"] / nt, 4) if nt > 0 else 0.0
        a["total_net_pnl"] = round(a["total_net_pnl"], 4)

    # Winrate per sisi (LONG/SHORT) untuk Kelly
    all_trades  = [t for f in ok_folds for t in f["trades"]]
    long_trades  = [t for t in all_trades if t["direction"] == "LONG"]
    short_trades = [t for t in all_trades if t["direction"] == "SHORT"]
    wr_long  = sum(1 for t in long_trades  if t["win"]) / len(long_trades)  if long_trades  else 0.0
    wr_short = sum(1 for t in short_trades if t["win"]) / len(short_trades) if short_trades else 0.0

    # "after" dict — backward compat dengan analyst.py yang baca bt_result["after"]
    after = {
        "winrate_up":  round(wr_long,  4),
        "winrate_dn":  round(wr_short, 4),
        "accuracy":    round(overall_wr, 4),
        "n_signal_up": sum(f["n_signal_up"] for f in ok_folds),
        "n_signal_dn": sum(f["n_signal_dn"] for f in ok_folds),
    }

    summary_text = _build_summary(
        symbol, interval, n_folds, ok_folds, total_trades, overall_wr,
        total_pnl, regime_agg, wr_long, wr_short,
    )

    logger.info(
        "[wfv] %s done — %d folds | WR=%.1f%% | PnL_net=$%.2f | trades=%d",
        symbol, n_folds, overall_wr * 100, total_pnl, total_trades,
    )

    return {
        "folds":        folds,
        "ok_folds":     ok_folds,
        "n_folds":      n_folds,
        "after":        after,          # backward compat
        "before":       after,          # placeholder (WFV tidak ada "before" weights)
        "regime_agg":   dict(regime_agg),
        "total_trades": total_trades,
        "total_wins":   total_wins,
        "overall_wr":   round(overall_wr, 4),
        "total_net_pnl":round(total_pnl,  4),
        "wr_long":      round(wr_long,  4),
        "wr_short":     round(wr_short, 4),
        "summary_text": summary_text,
        "sim_margin":   SIM_MARGIN,
        "sim_leverage": SIM_LEVERAGE,
        "train_window": TRAIN_WINDOW,
        "test_window":  TEST_WINDOW,
    }


# ------------------------------------------------------------------
# Summary text
# ------------------------------------------------------------------

def _build_summary(
    symbol, interval, n_folds, ok_folds,
    total_trades, overall_wr, total_pnl,
    regime_agg, wr_long, wr_short,
) -> str:
    pnl_sign = "+" if total_pnl >= 0 else ""
    lines = [
        f"Walk-Forward Validation — {symbol} ({interval})",
        f"  Config : train={TRAIN_WINDOW} | test={TEST_WINDOW} | step={STEP} | {n_folds} fold",
        f"  Modal  : ${SIM_MARGIN:.0f} per trade, leverage {SIM_LEVERAGE}x",
        f"  Overall: WR={overall_wr*100:.1f}% | PnL_net={pnl_sign}${total_pnl:.2f} | {total_trades} trades",
        f"  WR breakdown: LONG={wr_long*100:.1f}% | SHORT={wr_short*100:.1f}%",
        f"",
        f"  Per regime:",
    ]
    for reg, a in sorted(regime_agg.items()):
        nt   = a["total_trades"]
        wr   = a["winrate"] * 100
        pnl  = a["total_net_pnl"]
        psign= "+" if pnl >= 0 else ""
        prof = a["profitable_folds"]
        tot  = a["folds"]
        lines.append(
            f"    {reg:<10}: WR={wr:.1f}% | PnL={psign}${pnl:.2f} | "
            f"{nt} trades | {prof}/{tot} fold profit"
        )

    lines.append("")
    lines.append("  Fold detail:")
    for f in ok_folds:
        pnl_s = "+" if f["net_pnl"] >= 0 else ""
        lines.append(
            f"    Fold {f['fold']+1}: {f['regime']:<10} | "
            f"WR={f['winrate']*100:.1f}% | PnL={pnl_s}${f['net_pnl']:.2f} | "
            f"{f['n_trades']} trades"
        )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Format Telegram (2 pesan)
# ------------------------------------------------------------------

def format_telegram(symbol: str, wfv_result: dict, train_result: dict) -> list[str]:
    ra       = wfv_result["regime_agg"]
    wr_l     = wfv_result["wr_long"]   * 100
    wr_s     = wfv_result["wr_short"]  * 100
    ov_wr    = wfv_result["overall_wr"]* 100
    pnl      = wfv_result["total_net_pnl"]
    n_folds  = wfv_result["n_folds"]
    interval = train_result["interval"]
    n        = train_result["n_candles"]
    pnl_sign = "+" if pnl >= 0 else ""

    msg1_lines = [
        f"📊 <b>Walk-Forward Validation — {symbol} {interval} ({n} candles)</b>",
        f"   {n_folds} fold | train={TRAIN_WINDOW} | test={TEST_WINDOW} | step={STEP}",
        f"   Modal simulasi: ${SIM_MARGIN:.0f}/trade × {SIM_LEVERAGE}x leverage",
        f"─────────────────────────",
        f"",
        f"<b>Overall:</b>",
        f"  🎯 WR Overall : {ov_wr:.1f}%",
        f"  💹 WR LONG    : {wr_l:.1f}%",
        f"  💹 WR SHORT   : {wr_s:.1f}%",
        f"  💰 PnL Net    : {pnl_sign}${pnl:.2f}",
        f"  📦 Total Trade: {wfv_result['total_trades']}",
        f"",
        f"<b>Per Regime:</b>",
    ]

    regime_emoji = {"Trending": "📈", "Sideways": "↔️", "Volatile": "⚡"}
    for reg, a in sorted(ra.items()):
        emoji  = regime_emoji.get(reg, "•")
        nt     = a["total_trades"]
        wr     = a["winrate"] * 100
        p      = a["total_net_pnl"]
        ps     = "+" if p >= 0 else ""
        prof   = a["profitable_folds"]
        tot    = a["folds"]
        msg1_lines.append(
            f"  {emoji} <b>{reg:<10}</b> WR={wr:.1f}% | PnL={ps}${p:.2f} | "
            f"{nt} trades | {prof}/{tot} fold"
        )

    msg1 = "\n".join(msg1_lines)

    # Pesan 2: fold detail
    fold_lines = [f"📋 <b>Fold Detail — {symbol}</b>", "─────────────────────────"]
    for f in wfv_result["ok_folds"]:
        emoji  = regime_emoji.get(f["regime"], "•")
        ps     = "+" if f["net_pnl"] >= 0 else ""
        status = "✅" if f["net_pnl"] > 0 else "❌"
        fold_lines.append(
            f"  {status} Fold {f['fold']+1} {emoji} {f['regime']:<10} | "
            f"WR={f['winrate']*100:.1f}% | PnL={ps}${f['net_pnl']:.2f} | "
            f"{f['n_trades']} trades"
        )

    fold_lines.append("")
    fold_lines.append(
        f"<i>Fee: {FEE_RATE_RT*100:.1f}% RT | Slippage: live orderbook</i>"
    )
    msg2 = "\n".join(fold_lines)

    return [msg1, msg2]
