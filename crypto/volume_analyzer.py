import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

BINANCE_FUTURES_URL = "https://fapi.binance.com"

# Spike trigger
SPIKE_MULTIPLIER             = 6.5
BTC_SCANNER_SPIKE_MULTIPLIER = 2.0   
MIN_ATR_BUCKETS              = 30    

# Bucket = 1 menit (aggregasi volume per menit)
BUCKET_SEC = 60

# Fetch 1 hari data historis = 1440 menit
HISTORY_MINUTES = 1440


# ---------------------------------------------------------------------------
# Fetch historis aggTrades 1 hari via klines 1m
# ---------------------------------------------------------------------------

def _fetch_1d_klines(symbol: str) -> Optional[List[dict]]:
    """
    Fetch kline 1m 1 hari terakhir dari Binance Futures.
    Return list of dict dengan buy_vol dan sell_vol per menit.
    Binance kline menyediakan taker_buy_base_vol — kita pakai itu.
    """
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/klines",
            params={
                "symbol":   symbol.upper(),
                "interval": "1m",
                "limit":    HISTORY_MINUTES,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()

        buckets = []
        for c in raw:
            # c[5]  = volume (base)
            # c[9]  = taker_buy_base_asset_volume
            # c[4]  = close price (untuk konversi ke USDT)
            close_px  = float(c[4])
            total_vol = float(c[5]) * close_px           # total volume USDT
            buy_vol   = float(c[9]) * close_px           # taker buy volume USDT
            sell_vol  = total_vol - buy_vol              # sisanya = sell (maker buy)

            buckets.append({
                "open_time": int(c[0]),
                "buy_vol":   max(buy_vol,  0.0),
                "sell_vol":  max(sell_vol, 0.0),
                "total_vol": total_vol,
            })

        logger.info(
            "[vol] %s — fetched %d kline 1m (%.1f jam)",
            symbol, len(buckets), len(buckets) / 60,
        )
        return buckets

    except Exception as e:
        logger.warning("[vol] Gagal fetch klines %s: %s", symbol, e)
        return None


# ---------------------------------------------------------------------------
# ATR-style calculation (mean absolute deviation dari per-menit volume)
# ---------------------------------------------------------------------------

def _calc_atr(values: np.ndarray, period: int = 14) -> float:
    """
    Hitung ATR-style dari array volume per menit.
    Menggunakan Wilder smoothing (EMA dengan period).
    Ini bukan ATR price — tapi ATR volume (volatilitas volume per menit).
    """
    if len(values) < period + 1:
        return float(np.mean(values)) if len(values) > 0 else 0.0

    # True Range versi volume = abs(vol[i] - vol[i-1])
    tr = np.abs(np.diff(values))

    # Wilder smoothing
    atr = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period

    return atr


# ---------------------------------------------------------------------------
# State per simbol
# ---------------------------------------------------------------------------

@dataclass
class VolumeState:
    symbol:       str
    atr_buy:      float = 0.0    # ATR buy_vol per menit dari historis
    atr_sell:     float = 0.0    # ATR sell_vol per menit dari historis
    baseline_ok:  bool  = False  # True jika data historis cukup

    # Akumulasi volume menit berjalan (reset tiap menit)
    _bucket_start:    float = field(default=0.0,   repr=False)
    _cur_buy_vol:     float = field(default=0.0,   repr=False)
    _cur_sell_vol:    float = field(default=0.0,   repr=False)

    def feed_trade(self, price: float, qty: float, is_buyer_maker: bool, ts: float) -> None:
        """Terima satu aggTrade dari WebSocket, akumulasi ke bucket menit berjalan."""
        # Inisialisasi bucket pertama
        if self._bucket_start == 0.0:
            self._bucket_start = ts

        # Tutup bucket jika sudah >= 60 detik — reset akumulator
        if ts - self._bucket_start >= BUCKET_SEC:
            self._cur_buy_vol  = 0.0
            self._cur_sell_vol = 0.0
            self._bucket_start = ts

        vol_usdt = qty * price
        if is_buyer_maker:
            # is_buyer_maker=True → buyer adalah maker → inisiator = seller (market sell)
            self._cur_sell_vol += vol_usdt
        else:
            # is_buyer_maker=False → buyer adalah taker → inisiator = buyer (market buy)
            self._cur_buy_vol  += vol_usdt

    def current_buy_vol(self) -> float:
        return self._cur_buy_vol

    def current_sell_vol(self) -> float:
        return self._cur_sell_vol

    def buy_ratio(self) -> float:
        """Rasio buy_vol_sekarang / atr_buy. 0.0 jika baseline belum ready."""
        if not self.baseline_ok or self.atr_buy <= 0:
            return 0.0
        return self._cur_buy_vol / self.atr_buy

    def sell_ratio(self) -> float:
        """Rasio sell_vol_sekarang / atr_sell. 0.0 jika baseline belum ready."""
        if not self.baseline_ok or self.atr_sell <= 0:
            return 0.0
        return self._cur_sell_vol / self.atr_sell


# ---------------------------------------------------------------------------
# VolumeAnalyzer — interface utama yang dipakai monitor.py
# ---------------------------------------------------------------------------

class VolumeAnalyzer:

    def __init__(self, spike_multiplier: float = SPIKE_MULTIPLIER):
        self.spike_mult = spike_multiplier
        self._states: Dict[str, VolumeState] = {}

    def init_symbol(self, symbol: str) -> VolumeState:
        """
        Init state untuk simbol. Fetch data historis 1 hari langsung
        dan hitung ATR baseline buy/sell volume per menit.
        """
        sym = symbol.upper()
        if sym in self._states:
            return self._states[sym]

        state = VolumeState(symbol=sym)
        self._states[sym] = state

        buckets = _fetch_1d_klines(sym)
        if not buckets or len(buckets) < MIN_ATR_BUCKETS:
            logger.warning(
                "[vol] %s — data historis tidak cukup (%d bucket, butuh >= %d). "
                "Spike detection nonaktif sampai data tersedia.",
                sym, len(buckets) if buckets else 0, MIN_ATR_BUCKETS,
            )
            return state

        buy_vols  = np.array([b["buy_vol"]  for b in buckets], dtype=float)
        sell_vols = np.array([b["sell_vol"] for b in buckets], dtype=float)

        state.atr_buy  = _calc_atr(buy_vols)
        state.atr_sell = _calc_atr(sell_vols)
        state.baseline_ok = (state.atr_buy > 0 and state.atr_sell > 0)

        logger.info(
            "[vol] %s — baseline OK | atr_buy=%.0f USDT/min | atr_sell=%.0f USDT/min | "
            "spike_mult=%.0fx | trigger buy>=%.0f | trigger sell>=%.0f",
            sym,
            state.atr_buy,
            state.atr_sell,
            self.spike_mult,
            state.atr_buy  * self.spike_mult,
            state.atr_sell * self.spike_mult,
        )

        return state

    def remove_symbol(self, symbol: str) -> None:
        self._states.pop(symbol.upper(), None)

    def get_state(self, symbol: str) -> Optional[VolumeState]:
        return self._states.get(symbol.upper())

    def feed(self, symbol: str, price: float, qty: float, is_buyer_maker: bool) -> None:
        """Feed satu aggTrade ke state simbol."""
        state = self._states.get(symbol.upper())
        if state:
            state.feed_trade(price, qty, is_buyer_maker, time.time())

    def check_sell_spike(self, symbol: str) -> Tuple[bool, str]:
        state = self._states.get(symbol.upper())
        if state is None or not state.baseline_ok:
            return False, ""

        ratio   = state.sell_ratio()
        current = state.current_sell_vol()

        if ratio >= self.spike_mult:
            reason = (
                f"Sell spike {symbol}: {current:,.0f} USDT/min "
                f"= {ratio:.1f}× ATR ({state.atr_sell:,.0f})"
            )
            logger.info("[vol] SPIKE — %s", reason)
            return True, reason

        return False, ""

    def check_buy_spike(self, symbol: str) -> Tuple[bool, str]:
        state = self._states.get(symbol.upper())
        if state is None or not state.baseline_ok:
            return False, ""

        ratio   = state.buy_ratio()
        current = state.current_buy_vol()

        if ratio >= self.spike_mult:
            reason = (
                f"Buy spike {symbol}: {current:,.0f} USDT/min "
                f"= {ratio:.1f}× ATR ({state.atr_buy:,.0f})"
            )
            logger.info("[vol] SPIKE — %s", reason)
            return True, reason

        return False, ""
    
    def check_btc_scanner_spike(self) -> tuple:
        state = self._states.get("BTCUSDT")
        if state is None or not state.baseline_ok:
            return False, "", ""

        buy_ratio  = state.buy_ratio()
        sell_ratio = state.sell_ratio()

        if buy_ratio >= BTC_SCANNER_SPIKE_MULTIPLIER:
            current = state.current_buy_vol()
            reason  = (
                f"BTC buy spike: {current:,.0f} USDT/min "
                f"= {buy_ratio:.1f}× ATR ({state.atr_buy:,.0f})"
            )
            logger.warning("[vol] BTC SCANNER SPIKE (buy) — %s", reason)
            return True, "buy", reason

        if sell_ratio >= BTC_SCANNER_SPIKE_MULTIPLIER:
            current = state.current_sell_vol()
            reason  = (
                f"BTC sell spike: {current:,.0f} USDT/min "
                f"= {sell_ratio:.1f}× ATR ({state.atr_sell:,.0f})"
            )
            logger.warning("[vol] BTC SCANNER SPIKE (sell) — %s", reason)
            return True, "sell", reason

        return False, "", ""


    def get_status(self, symbol: str) -> dict:
        """Debug info untuk satu simbol."""
        state = self._states.get(symbol.upper())
        if state is None:
            return {"symbol": symbol, "error": "not initialized"}
        return {
            "symbol":       symbol.upper(),
            "baseline_ok":  state.baseline_ok,
            "atr_buy":      round(state.atr_buy,         2),
            "atr_sell":     round(state.atr_sell,        2),
            "cur_buy_vol":  round(state.current_buy_vol(),  2),
            "cur_sell_vol": round(state.current_sell_vol(), 2),
            "buy_ratio":    round(state.buy_ratio(),     2),
            "sell_ratio":   round(state.sell_ratio(),    2),
            "spike_mult":   self.spike_mult,
            "buy_trigger":  round(state.atr_buy  * self.spike_mult, 2),
            "sell_trigger": round(state.atr_sell * self.spike_mult, 2),
        }
