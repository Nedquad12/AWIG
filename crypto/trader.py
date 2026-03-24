"""
trader.py — Eksekusi order + pasang SL/TP setelah entry

Flow:
  1. Terima keputusan AI (decision, direction, leverage, capital_pct, sl_pct, tp_pct)
  2. Hitung quantity dari balance * capital_pct * leverage / mark_price
  3. Set leverage
  4. Place market order (entry)
  5. Hitung harga SL dan TP dari entry price + sl_pct/tp_pct
  6. Place STOP_MARKET order untuk SL
  7. Place TAKE_PROFIT_MARKET order untuk TP
  8. Simpan semua info posisi ke positions.json

Cara hitung harga SL/TP:
  LONG : SL = entry * (1 - sl_pct/100),  TP = entry * (1 + tp_pct/100)
  SHORT: SL = entry * (1 + sl_pct/100),  TP = entry * (1 - tp_pct/100)
"""

import json
import logging
import math
import os
from datetime import datetime
from typing import Optional

from api_binance import (
    get_account_balance, get_mark_price, get_exchange_info,
    set_leverage, place_market_order, place_stop_order,
    cancel_open_orders, get_open_positions,
)
from config import POSITIONS_FILE

logger = logging.getLogger(__name__)


# ── Positions state ───────────────────────────────────────────────────────────

def load_positions() -> dict:
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_positions(positions: dict):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


# ── Quantity & price helpers ──────────────────────────────────────────────────

def _calc_step_size(symbol: str) -> float:
    """Ambil minimum step size dari exchange info."""
    info = get_exchange_info(symbol)
    for filt in info.get("filters", []):
        if filt.get("filterType") == "LOT_SIZE":
            return float(filt.get("stepSize", "0.001"))
    return 0.001


def _calc_tick_size(symbol: str) -> float:
    """Ambil tick size (presisi harga) dari exchange info."""
    info = get_exchange_info(symbol)
    for filt in info.get("filters", []):
        if filt.get("filterType") == "PRICE_FILTER":
            return float(filt.get("tickSize", "0.01"))
    return 0.01


def _round_to_step(qty: float, step: float) -> float:
    """Round quantity ke step size yang valid (floor)."""
    if step <= 0:
        return qty
    precision = max(0, -int(math.floor(math.log10(step))))
    return round(math.floor(qty / step) * step, precision)


def _round_price(price: float, tick: float) -> float:
    """Round harga ke tick size yang valid."""
    if tick <= 0:
        return price
    precision = max(0, -int(math.floor(math.log10(tick))))
    return round(round(price / tick) * tick, precision)


def calculate_quantity(
    symbol:      str,
    capital_pct: float,
    leverage:    int,
) -> tuple[float, float]:
    """
    Hitung jumlah kontrak yang bisa dibeli.

    Returns:
        (quantity, mark_price)
    """
    balance    = get_account_balance("USDT")
    mark_price = get_mark_price(symbol)

    if balance <= 0 or mark_price <= 0:
        return 0.0, mark_price

    allocated  = balance * capital_pct * leverage
    step_size  = _calc_step_size(symbol)
    raw_qty    = allocated / mark_price
    quantity   = _round_to_step(raw_qty, step_size)

    logger.info(
        f"[{symbol}] balance={balance:.2f} USDT | "
        f"capital={capital_pct*100:.0f}% | lev={leverage}x | "
        f"mark={mark_price:.4f} | qty={quantity}"
    )
    return quantity, mark_price


def calc_sl_tp_prices(
    entry_price: float,
    direction:   str,    # "LONG" atau "SHORT"
    sl_pct:      float,  # % positif
    tp_pct:      float,  # % positif
    tick_size:   float = 0.01,
) -> tuple[float, float]:
    """
    Hitung harga absolut SL dan TP dari entry price dan persentase.

    Returns:
        (sl_price, tp_price)
    """
    if direction == "LONG":
        sl_price = entry_price * (1 - sl_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100)
    else:  # SHORT
        sl_price = entry_price * (1 + sl_pct / 100)
        tp_price = entry_price * (1 - tp_pct / 100)

    sl_price = _round_price(sl_price, tick_size)
    tp_price = _round_price(tp_price, tick_size)

    return sl_price, tp_price


# ── Main execute function ─────────────────────────────────────────────────────

def execute_trade(symbol: str, decision: dict) -> dict:
    """
    Eksekusi trade berdasarkan keputusan AI.
    Setelah entry berhasil, langsung pasang SL dan TP order.

    Args:
        symbol  : e.g. "BTCUSDT"
        decision: output dari ai_decision.ask_ai()
                  harus mengandung: decision, direction, leverage,
                  capital_pct, sl_pct, tp_pct

    Returns:
        dict hasil eksekusi lengkap
    """
    if decision.get("decision") != "BUY":
        return {
            "success": False,
            "symbol":  symbol,
            "message": f"SKIP — {decision.get('reason', '')}",
        }

    direction   = decision.get("direction", "LONG")
    leverage    = int(decision.get("leverage", 1))
    capital_pct = float(decision.get("capital_pct", 0.05))
    sl_pct      = float(decision.get("sl_pct", 1.0))
    tp_pct      = float(decision.get("tp_pct", 2.0))

    # ── 1. Hitung quantity ────────────────────────────────────────────────
    quantity, entry_price = calculate_quantity(symbol, capital_pct, leverage)
    if quantity <= 0:
        return {
            "success": False,
            "symbol":  symbol,
            "message": "Quantity = 0. Balance tidak cukup atau mark price error.",
        }

    # ── 2. Set leverage ───────────────────────────────────────────────────
    if not set_leverage(symbol, leverage):
        return {
            "success": False,
            "symbol":  symbol,
            "message": f"Gagal set leverage {leverage}x untuk {symbol}",
        }

    # ── 3. Entry order ────────────────────────────────────────────────────
    entry_side = "BUY" if direction == "LONG" else "SELL"
    order = place_market_order(symbol, entry_side, quantity)
    if order is None:
        return {
            "success": False,
            "symbol":  symbol,
            "message": "Gagal place entry order ke Binance.",
        }

    # Ambil actual fill price jika tersedia dari order response
    avg_price = float(order.get("avgPrice") or entry_price)
    if avg_price <= 0:
        avg_price = entry_price

    # ── 4. Hitung harga SL & TP ───────────────────────────────────────────
    tick_size        = _calc_tick_size(symbol)
    sl_price, tp_price = calc_sl_tp_prices(avg_price, direction, sl_pct, tp_pct, tick_size)

    # Sisi closing order: kebalikan dari entry
    close_side = "SELL" if direction == "LONG" else "BUY"

    # ── 5. Place Stop Loss ────────────────────────────────────────────────
    sl_order = place_stop_order(
        symbol     = symbol,
        side       = close_side,
        quantity   = quantity,
        stop_price = sl_price,
        order_type = "STOP_MARKET",
    )
    sl_order_id = sl_order.get("orderId") if sl_order else None
    if sl_order is None:
        logger.warning(f"[{symbol}] SL order gagal dipasang! Posisi terbuka tanpa SL.")

    # ── 6. Place Take Profit ──────────────────────────────────────────────
    tp_order = place_stop_order(
        symbol     = symbol,
        side       = close_side,
        quantity   = quantity,
        stop_price = tp_price,
        order_type = "TAKE_PROFIT_MARKET",
    )
    tp_order_id = tp_order.get("orderId") if tp_order else None
    if tp_order is None:
        logger.warning(f"[{symbol}] TP order gagal dipasang! Posisi terbuka tanpa TP.")

    # ── 7. Simpan state posisi ────────────────────────────────────────────
    positions = load_positions()
    positions[symbol] = {
        "symbol":        symbol,
        "direction":     direction,
        "side":          entry_side,
        "quantity":      quantity,
        "entry_price":   avg_price,
        "leverage":      leverage,
        "capital_pct":   capital_pct,
        "sl_pct":        sl_pct,
        "tp_pct":        tp_pct,
        "sl_price":      sl_price,
        "tp_price":      tp_price,
        "entry_order_id": order.get("orderId"),
        "sl_order_id":   sl_order_id,
        "tp_order_id":   tp_order_id,
        "opened_at":     datetime.utcnow().isoformat(),
        "ai_reason":     decision.get("reason", ""),
        "ai_confidence": decision.get("confidence", 0),
    }
    save_positions(positions)

    logger.info(
        f"[{symbol}] Trade opened | {direction} qty={quantity} @ {avg_price:.4f} | "
        f"SL={sl_price:.4f} ({sl_pct}%) | TP={tp_price:.4f} ({tp_pct}%) | "
        f"lev={leverage}x"
    )

    return {
        "success":       True,
        "symbol":        symbol,
        "direction":     direction,
        "side":          entry_side,
        "quantity":      quantity,
        "entry_price":   avg_price,
        "leverage":      leverage,
        "capital_pct":   capital_pct,
        "sl_pct":        sl_pct,
        "tp_pct":        tp_pct,
        "sl_price":      sl_price,
        "tp_price":      tp_price,
        "sl_order_id":   sl_order_id,
        "tp_order_id":   tp_order_id,
        "entry_order_id": order.get("orderId"),
        "message": (
            f"{direction} {quantity} {symbol} @ {avg_price:.4f} | "
            f"{leverage}x lev | SL {sl_price:.4f} ({sl_pct}%) | TP {tp_price:.4f} ({tp_pct}%)"
        ),
    }


# ── Close position ────────────────────────────────────────────────────────────

def close_trade(symbol: str, reason: str = "manual") -> dict:
    """
    Tutup posisi yang ada untuk symbol.

    Flow:
      1. Cancel semua open orders SL/TP terlebih dahulu
      2. Market order berlawanan untuk flat posisi
      3. Hapus dari positions.json

    Args:
        symbol : e.g. "BTCUSDT"
        reason : alasan penutupan (untuk logging)
    """
    # Cancel SL/TP yang masih tergantung
    cancel_open_orders(symbol)

    # Ambil posisi aktual dari Binance
    open_positions = get_open_positions()
    target = next((p for p in open_positions if p["symbol"] == symbol), None)

    if target is None:
        # Posisi sudah tidak ada (mungkin sudah kena SL/TP), bersihkan local state
        positions = load_positions()
        local = positions.pop(symbol, None)
        save_positions(positions)
        return {
            "success": False,
            "symbol":  symbol,
            "message": f"Tidak ada posisi terbuka untuk {symbol} (sudah closed/SL/TP?)",
        }

    close_side = "SELL" if target["side"] == "LONG" else "BUY"
    quantity   = abs(target["positionAmt"])

    order = place_market_order(symbol, close_side, quantity)
    if order is None:
        return {"success": False, "symbol": symbol, "message": "Gagal place close order"}

    # Hitung PnL kasar dari local state
    positions = load_positions()
    local     = positions.pop(symbol, {})
    save_positions(positions)

    entry  = local.get("entry_price", 0)
    exit_p = float(order.get("avgPrice") or target.get("entryPrice", 0))
    pnl_pct = 0.0
    if entry > 0 and exit_p > 0:
        direction = local.get("direction", "LONG")
        raw_pnl   = (exit_p - entry) / entry * 100
        pnl_pct   = raw_pnl if direction == "LONG" else -raw_pnl
        pnl_pct  *= local.get("leverage", 1)

    logger.info(
        f"[{symbol}] Position closed ({reason}) | {close_side} qty={quantity} | "
        f"entry={entry:.4f} exit={exit_p:.4f} | est PnL={pnl_pct:+.2f}%"
    )

    return {
        "success":    True,
        "symbol":     symbol,
        "side":       close_side,
        "quantity":   quantity,
        "entry_price": entry,
        "exit_price":  exit_p,
        "pnl_pct":    round(pnl_pct, 2),
        "order_id":   order.get("orderId"),
        "reason":     reason,
        "message":    f"Closed {symbol} {close_side} qty={quantity} @ ~{exit_p:.4f} | est PnL {pnl_pct:+.2f}%",
    }
