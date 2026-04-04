import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

BINANCE_FUTURES_URL = "https://fapi.binance.com"

BUCKET_SEC     = 60
ATR_BUCKETS    = 14
SPIKE_MULT     = 10.0
MIN_BUCKETS    = 5


@dataclass
class VolumeBucket:
    ts_start:   float
    buy_vol:    float = 0.0   
    sell_vol:   float = 0.0  
    trade_count: int  = 0

    @property
    def total_vol(self) -> float:
        return self.buy_vol + self.sell_vol


@dataclass
class VolumeState:
    symbol:        str
    window_size:   int                     
    buckets:       Deque[VolumeBucket] = field(default_factory=lambda: deque(maxlen=ATR_BUCKETS + 5))
    _current_bucket: Optional[VolumeBucket]  = field(default=None, repr=False)
    initialized:   bool = False

    def feed_trade(self, price: float, qty: float, is_buyer_maker: bool, ts: float) -> None:
        if self._current_bucket is None:
            self._current_bucket = VolumeBucket(ts_start=ts)
        if ts - self._current_bucket.ts_start >= BUCKET_SEC:
            self.buckets.append(self._current_bucket)
            self._current_bucket = VolumeBucket(ts_start=ts)

        vol_usdt = qty * price  
        if is_buyer_maker:
            self._current_bucket.sell_vol += vol_usdt
        else:
            self._current_bucket.buy_vol += vol_usdt
        self._current_bucket.trade_count += 1

    def get_sell_volumes(self) -> List[float]:
        return [b.sell_vol for b in self.buckets]

    def get_buy_volumes(self) -> List[float]:
        return [b.buy_vol for b in self.buckets]

    def current_sell_vol(self) -> float:
        return self._current_bucket.sell_vol if self._current_bucket else 0.0

    def current_buy_vol(self) -> float:
        return self._current_bucket.buy_vol if self._current_bucket else 0.0

    def bucket_count(self) -> int:
        return len(self.buckets)

def _calc_volume_atr(volumes: List[float], period: int = ATR_BUCKETS) -> Optional[float]:

    if len(volumes) < MIN_BUCKETS:
        return None

    arr = np.array(volumes[-period:], dtype=float)
    avg = float(np.mean(arr))
    return avg if avg > 0 else None

def fetch_daily_trade_count(symbol: str) -> int:
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr",
            params={"symbol": symbol.upper()},
            timeout=5,
        )
        resp.raise_for_status()
        count = int(resp.json().get("count", 10_000))
        logger.info("[vol] %s daily trade count: %d", symbol, count)
        return count
    except Exception as e:
        logger.warning("[vol] Gagal fetch trade count %s: %s — pakai default 10000", symbol, e)
        return 10_000

def calc_window_size(symbol: str) -> int:
    """Window = 25% dari daily trade count."""
    count = fetch_daily_trade_count(symbol)
    return max(int(count * 0.25), 500)   

class VolumeAnalyzer:

    def __init__(self, spike_multiplier: float = SPIKE_MULT):
        self.spike_mult = spike_multiplier
        self._states: dict[str, VolumeState] = {}

    def init_symbol(self, symbol: str) -> VolumeState:
  
        sym = symbol.upper()
        if sym not in self._states:
            window = calc_window_size(sym)
            self._states[sym] = VolumeState(
                symbol      = sym,
                window_size = window,
                buckets     = deque(maxlen=ATR_BUCKETS + 5),
            )
            logger.info("[vol] Init %s — window_size=%d trades", sym, window)
        return self._states[sym]

    def remove_symbol(self, symbol: str) -> None:
        self._states.pop(symbol.upper(), None)

    def get_state(self, symbol: str) -> Optional[VolumeState]:
        return self._states.get(symbol.upper())

    def feed(self, symbol: str, price: float, qty: float, is_buyer_maker: bool) -> None:
        state = self._states.get(symbol.upper())
        if state:
            state.feed_trade(price, qty, is_buyer_maker, time.time())

    def check_sell_spike(self, symbol: str) -> Tuple[bool, str]:
        state = self._states.get(symbol.upper())
        if state is None or state.bucket_count() < MIN_BUCKETS:
            return False, ""

        sell_vols = state.get_sell_volumes()
        atr       = _calc_volume_atr(sell_vols)
        if atr is None or atr == 0:
            return False, ""

        current = state.current_sell_vol()
        ratio   = current / atr

        if ratio >= self.spike_mult:
            reason = (
                f"Sell spike {symbol}: {current:,.0f} USDT "
                f"= {ratio:.1f}× ATR ({atr:,.0f})"
            )
            logger.info("[vol] %s", reason)
            return True, reason

        return False, ""

    def check_buy_spike(self, symbol: str) -> Tuple[bool, str]:
        state = self._states.get(symbol.upper())
        if state is None or state.bucket_count() < MIN_BUCKETS:
            return False, ""

        buy_vols = state.get_buy_volumes()
        atr      = _calc_volume_atr(buy_vols)
        if atr is None or atr == 0:
            return False, ""

        current = state.current_buy_vol()
        ratio   = current / atr

        if ratio >= self.spike_mult:
            reason = (
                f"Buy spike {symbol}: {current:,.0f} USDT "
                f"= {ratio:.1f}× ATR ({atr:,.0f})"
            )
            logger.info("[vol] %s", reason)
            return True, reason

        return False, ""
