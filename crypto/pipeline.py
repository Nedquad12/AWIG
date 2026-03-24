"""
pipeline.py — Orkestrasi full flow untuk 1 symbol

Flow:
  fetch OHLCV → hitung skor → build history → ML adjust → AI decide → execute
"""

import logging
from typing import Optional

import pandas as pd

from api_binance import fetch_ohlcv, get_account_balance, get_open_positions
from scorer import calculate_all_scores
from history_manager import build_score_history, save_history, load_history
from ml_engine import run_ml_weights, predict_direction
from ai_decision import ask_ai
from trader import execute_trade
from config import SCORE_WARMUP

logger = logging.getLogger(__name__)


def run_full_pipeline(
    symbol:        str,
    execute:       bool = True,    # False = dry run (tidak eksekusi)
    skip_ml:       bool = False,   # True = skip ML weight update
) -> dict:
    """
    Jalankan full pipeline untuk 1 symbol.

    Returns dict berisi semua hasil per tahap.
    """
    result = {
        "symbol":   symbol,
        "stage":    "init",
        "fetch":    None,
        "scores":   None,
        "ml_adj":   None,
        "ml_pred":  None,
        "ai":       None,
        "trade":    None,
        "error":    None,
    }

    # ── TAHAP 1: Fetch OHLCV ──────────────────────────────────────────────
    result["stage"] = "fetch"
    df = fetch_ohlcv(symbol)
    if df is None or len(df) < SCORE_WARMUP + 10:
        result["error"] = f"Data tidak cukup untuk {symbol} (min {SCORE_WARMUP+10} candle)"
        return result
    result["fetch"] = {"candles": len(df), "last": str(df["date"].iloc[-1])}

    # ── TAHAP 2: Hitung Skor ──────────────────────────────────────────────
    result["stage"] = "score"
    try:
        scores = calculate_all_scores(symbol, df)
        result["scores"] = scores
    except Exception as e:
        result["error"] = f"Error scoring: {e}"
        return result

    # ── TAHAP 3: Build & simpan history (untuk ML) ────────────────────────
    result["stage"] = "history"
    try:
        history = build_score_history(symbol, df)
        save_history(symbol, history)
        logger.info(f"[{symbol}] History saved: {len(history)} bars")
    except Exception as e:
        logger.warning(f"[{symbol}] History error (non-fatal): {e}")

    # ── TAHAP 4: ML Weight Adjustment ─────────────────────────────────────
    result["stage"] = "ml_adj"
    if not skip_ml:
        try:
            ml_adj = run_ml_weights(symbol)
            result["ml_adj"] = ml_adj
            if ml_adj.get("success"):
                logger.info(
                    f"[{symbol}] ML adj: acc {ml_adj['accuracy_before']}% → {ml_adj['accuracy_after']}%"
                )
        except Exception as e:
            logger.warning(f"[{symbol}] ML adj error (non-fatal): {e}")
            result["ml_adj"] = {"success": False, "message": str(e)}
    else:
        result["ml_adj"] = {"success": False, "message": "skip_ml=True"}

    # ── TAHAP 5: ML Prediction ────────────────────────────────────────────
    result["stage"] = "ml_pred"
    try:
        ml_pred = predict_direction(symbol)
        result["ml_pred"] = ml_pred
    except Exception as e:
        logger.warning(f"[{symbol}] ML pred error (non-fatal): {e}")
        result["ml_pred"] = {"error": str(e)}

    # ── TAHAP 6: AI Decision ──────────────────────────────────────────────
    result["stage"] = "ai"
    try:
        balance        = get_account_balance("USDT")
        open_positions = get_open_positions()

        ai_decision = ask_ai(
            symbol         = symbol,
            scores         = scores,
            ml_pred        = result["ml_pred"] or {},
            df_recent      = df,
            balance        = balance,
            open_positions = open_positions,
        )
        result["ai"] = ai_decision

        if ai_decision.get("error"):
            result["error"] = f"AI error: {ai_decision['error']}"
            return result

    except Exception as e:
        result["error"] = f"AI exception: {e}"
        return result

    # ── TAHAP 7: Execute Trade ─────────────────────────────────────────────
    result["stage"] = "trade"
    if execute:
        try:
            trade_result = execute_trade(symbol, ai_decision)
            result["trade"] = trade_result
        except Exception as e:
            result["error"] = f"Trade exception: {e}"
            return result
    else:
        result["trade"] = {"message": "DRY RUN — tidak ada order dieksekusi"}

    result["stage"] = "done"
    return result


def run_scores_only(symbol: str) -> dict:
    """Hanya fetch + hitung skor, tanpa ML/AI/trade. Untuk /skor command."""
    df = fetch_ohlcv(symbol)
    if df is None or len(df) < 35:
        return {"error": f"Data tidak cukup untuk {symbol}"}
    try:
        return calculate_all_scores(symbol, df)
    except Exception as e:
        return {"error": str(e)}
