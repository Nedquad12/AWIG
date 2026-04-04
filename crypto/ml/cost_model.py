import logging
import time
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)

FEE_RATE_EACH_SIDE = 0.001          
FEE_RATE_RT        = FEE_RATE_EACH_SIDE * 2   

ORDERBOOK_DEPTH    = 20            
ORDERBOOK_CACHE_S  = 30           
FALLBACK_SLIPPAGE  = 0.0005       

_BINANCE_BASE = "https://fapi.binance.com"




_ob_cache: dict[str, tuple[float, dict]] = {}   # symbol → (timestamp, data)


def _fetch_orderbook(symbol: str, limit: int = ORDERBOOK_DEPTH) -> dict | None:
    """
    Fetch orderbook Binance Futures (fapi).
    Return {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
    Cache 30 detik per symbol.
    """
    now = time.time()
    if symbol in _ob_cache:
        ts, data = _ob_cache[symbol]
        if now - ts < ORDERBOOK_CACHE_S:
            return data

    try:
        url  = f"{_BINANCE_BASE}/fapi/v1/depth"
        resp = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        # Parse ke float
        parsed = {
            "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
            "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
        }
        _ob_cache[symbol] = (now, parsed)
        return parsed
    except Exception as e:
        logger.warning("[cost_model] Orderbook fetch gagal %s: %s", symbol, e)
        return None



def estimate_slippage(
    symbol: str,
    side: str,          
    notional_usdt: float, 
) -> float:
    ob = _fetch_orderbook(symbol)
    if ob is None:
        logger.debug("[cost_model] Pakai fallback slippage untuk %s", symbol)
        return FALLBACK_SLIPPAGE

    bids = ob["bids"]  
    asks = ob["asks"] 

    if not bids or not asks:
        return FALLBACK_SLIPPAGE

    mid_price = (bids[0][0] + asks[0][0]) / 2.0
    levels    = asks if side == "BUY" else bids

    remaining   = notional_usdt
    total_cost  = 0.0
    total_qty   = 0.0

    for price, qty in levels:
        level_notional = price * qty
        if level_notional >= remaining:
            fill_qty    = remaining / price
            total_cost += fill_qty * price
            total_qty  += fill_qty
            remaining   = 0
            break
        else:
            total_cost += level_notional
            total_qty  += qty
            remaining  -= level_notional

    if remaining > 0:
        logger.warning(
            "[cost_model] %s order size %.2f USDT melebihi orderbook depth. "
            "Slippage estimate tidak akurat.",
            symbol, notional_usdt,
        )
        if levels:
            last_price  = levels[-1][0]
            slippage    = abs(last_price - mid_price) / mid_price
            return min(slippage, 0.005) 
        return FALLBACK_SLIPPAGE

    if total_qty <= 0:
        return FALLBACK_SLIPPAGE

    vwap     = total_cost / total_qty
    slippage = abs(vwap - mid_price) / mid_price
    return round(slippage, 6)

def compute_trade_pnl(
    symbol: str,
    direction: str,      
    entry_price: float,
    exit_price: float,
    margin_usdt: float,  
    leverage: int,
    fetch_slippage: bool = True,
) -> dict:
    notional    = margin_usdt * leverage
    direction_s = 1 if direction == "LONG" else -1

    return_pct = (exit_price - entry_price) / entry_price * direction_s
    gross_pnl  = return_pct * notional
    fee_usdt = FEE_RATE_RT * notional

    if fetch_slippage:
        entry_side = "BUY"  if direction == "LONG" else "SELL"
        exit_side  = "SELL" if direction == "LONG" else "BUY"
        slip_entry = estimate_slippage(symbol, entry_side, notional)
        slip_exit  = estimate_slippage(symbol, exit_side,  notional)
        slippage_pct = slip_entry + slip_exit
    else:
        slippage_pct = FALLBACK_SLIPPAGE * 2

    slippage_usdt = slippage_pct * notional
    net_pnl       = gross_pnl - fee_usdt - slippage_usdt
    net_pnl_pct   = (net_pnl / margin_usdt) * 100 if margin_usdt > 0 else 0.0

    return {
        "gross_pnl":    round(gross_pnl,    4),
        "fee_usdt":     round(fee_usdt,     4),
        "slippage_pct": round(slippage_pct, 6),
        "slippage_usdt":round(slippage_usdt,4),
        "net_pnl":      round(net_pnl,      4),
        "net_pnl_pct":  round(net_pnl_pct,  4),
        "notional":     round(notional,     4),
        "return_pct":   round(return_pct,   6),
    }

def summarize_costs(trades: list[dict]) -> dict:
    if not trades:
        return {
            "n_trades": 0,
            "total_gross_pnl": 0.0,
            "total_fee": 0.0,
            "total_slippage": 0.0,
            "total_net_pnl": 0.0,
            "avg_slippage_pct": 0.0,
        }

    return {
        "n_trades":         len(trades),
        "total_gross_pnl":  round(sum(t["gross_pnl"]     for t in trades), 4),
        "total_fee":        round(sum(t["fee_usdt"]       for t in trades), 4),
        "total_slippage":   round(sum(t["slippage_usdt"]  for t in trades), 4),
        "total_net_pnl":    round(sum(t["net_pnl"]        for t in trades), 4),
        "avg_slippage_pct": round(
            sum(t["slippage_pct"] for t in trades) / len(trades), 6
        ),
    }
