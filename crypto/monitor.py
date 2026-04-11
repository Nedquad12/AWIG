# =============================================================
# monitor.py — Real-Time Position Monitor (LIVE ONLY)
#
# Tanggung jawab:
#   1. Baca posisi aktif dari positions.json
#   2. Subscribe WS aggTrade per koin + BTCUSDT
#   3. User Data Stream (WebSocket) untuk order fill/cancel real-time
#      → FILLED: aktifkan posisi di JSON + notif
#      → CANCELED/EXPIRED/REJECTED: cancel di JSON + notif
#   4. Per tick WS:
#       a. Evaluate posisi open (SL/TP/breakeven/trailing)
#       b. Partial TP @ RR 1.5 via Binance reduce-only market
#   5. Per bucket 60 detik: volume reversal koin + BTC
#   6. Per bucket 15 detik: Urgent Circuit Breaker BTC ≥ 5%
#   7. Semua close via WsOrderManager (WebSocket API) + REST fallback
# =============================================================

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set

import requests
import websockets
from websockets.exceptions import ConnectionClosed

from volume_analyzer import VolumeAnalyzer
from wti_crypto      import get_wti
from risk_manager    import register_sl, register_urgent_cb, is_urgent_cb_triggered

logger = logging.getLogger(__name__)

TOPIC_CB      = 6
TOPIC_ORDERS  = 2
TOPIC_ERROR   = 10
TOPIC_GENERAL = 88

BINANCE_WS_BASE     = "wss://fstream.binance.com/stream"
BINANCE_WS_API_URL  = "wss://ws-fapi.binance.com/ws-fapi/v1"   # WebSocket API (order execution)
BINANCE_FUTURES_URL = "https://fapi.binance.com"

RECONNECT_BASE = 2
RECONNECT_MAX  = 30

BREAKEVEN_RR          = 1.0
PENDING_EXPIRE_SEC    = 1 * 3600
MAX_HOLD_SEC          = 35 * 3600
MAX_FILLS_PER_SESSION = 2
URGENT_CB_PCT         = 0.04
URGENT_CB_BUCKET_SEC  = 15
TAKER_FEE             = 0.0004
PARTIAL_TP_RR         = 1.5
PARTIAL_TP_PCT        = 0.30

STALE_MOMENTUM_MIN_AGE_SEC  = 12 * 3600   # 12 jam minimal hold sebelum cek
STALE_MOMENTUM_CUT_PCT      = 0.25        # cut 40% dari qty_original
STALE_MOMENTUM_ATR_PERIOD   = 14          # ATR-14 dari 4H
STALE_MOMENTUM_ATR_MULT     = 3.2        # threshold = 2.5× ATR
STALE_MOMENTUM_MIN_MOVE_PCT = 0.012        # floor minimum 1%


# ------------------------------------------------------------------
# File I/O
# ------------------------------------------------------------------

def _find_file(name: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, name)
    return path


POSITIONS_FILE = _find_file("positions.json")
HISTORY_FILE   = _find_file("positions_history.json")


def _load_all() -> list:
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_all(data: list) -> None:
    with open(POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _append_history(record: dict) -> None:
    try:
        hist = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                hist = json.load(f)
        hist.append(record)
        with open(HISTORY_FILE, "w") as f:
            json.dump(hist, f, indent=2)
    except Exception as e:
        logger.error("[monitor] append_history error: %s", e)


def load_open_positions() -> list:
    return [p for p in _load_all() if p.get("status") == "open"]


def load_pending_positions() -> list:
    return [p for p in _load_all() if p.get("status") == "pending"]


# ------------------------------------------------------------------
# Binance helpers
# ------------------------------------------------------------------

def get_mark_price(symbol: str) -> Optional[float]:
    try:
        r = requests.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex",
            params={"symbol": symbol}, timeout=5,
        )
        return float(r.json().get("markPrice", 0)) or None
    except Exception:
        return None


def get_tick_size(symbol: str) -> float:
    try:
        from order.executor import get_symbol_info
        return get_symbol_info(symbol)["price_tick"]
    except Exception:
        return 0.0001


def get_qty_step(symbol: str) -> float:
    """Ambil LOT_SIZE stepSize dari Binance. Dipakai untuk floor qty sebelum order."""
    try:
        from order.executor import get_symbol_info
        return get_symbol_info(symbol)["qty_step"]
    except Exception:
        return 0.001


def floor_qty(qty: float, step: float) -> float:
    """Floor qty ke stepSize Binance (ROUND_DOWN, bukan round)."""
    if step <= 0:
        return qty
    from decimal import Decimal, ROUND_DOWN
    d_qty  = Decimal(str(qty))
    d_step = Decimal(str(step))
    return float((d_qty // d_step) * d_step)


def fetch_btc_prev_close() -> float:
    try:
        r = requests.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/klines",
            params={"symbol": "BTCUSDT", "interval": "1d", "limit": 2},
            timeout=10,
        )
        klines = r.json()
        if len(klines) >= 2:
            return float(klines[-2][4])
    except Exception as e:
        logger.warning("[monitor] fetch_btc_prev_close error: %s", e)
    return 0.0


def _fetch_4h_klines(symbol: str, limit: int = 20) -> list:
    """
    Ambil klines 4H dari Binance Futures.
    Kline terakhir dalam list = candle yang sedang berjalan (belum closed).
    """
    try:
        r = requests.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/klines",
            params={"symbol": symbol, "interval": "4h", "limit": limit},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("[monitor] _fetch_4h_klines %s error: %s", symbol, e)
        return []


def _calc_atr(klines: list, period: int = 14) -> float:
    """
    Hitung ATR(period) dari list klines Binance menggunakan SMA method.
    Return ATR dalam unit harga, atau 0.0 jika data tidak cukup.
    """
    if len(klines) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(klines)):
        high       = float(klines[i][2])
        low        = float(klines[i][3])
        prev_close = float(klines[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    atr_slice = trs[-period:]
    return sum(atr_slice) / len(atr_slice) if atr_slice else 0.0


# ------------------------------------------------------------------
# Pending management
# ------------------------------------------------------------------

def cancel_expired_pending(notify_fn=None) -> list:
    def _n(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    cancelled = []
    try:
        all_pos = _load_all()
        now, changed = time.time(), False
        for i, pos in enumerate(all_pos):
            if pos.get("status") != "pending":
                continue
            age = now - float(pos.get("opened_at", now))
            if age < PENDING_EXPIRE_SEC:
                continue
            all_pos[i]["status"]        = "cancelled"
            all_pos[i]["cancel_reason"] = f"expired — tidak ter-fill dalam {int(age/60)} menit"
            all_pos[i]["cancelled_at"]  = int(now)
            changed = True
            sym = pos.get("symbol", "?")
            cancelled.append(sym)
            try:
                from order.executor import cancel_all_open_orders
                cancel_all_open_orders(sym)
            except Exception as e:
                logger.warning("[monitor] cancel Binance expired %s: %s", sym, e)
            _n(
                f"⏱ <b>Order Expired — {sym}</b>\n"
                f"  Entry  : <code>{pos.get('entry_price')}</code>\n"
                f"  Side   : <b>{pos.get('side')}</b>\n"
                f"  Reason : tidak ter-fill dalam 2 jam\n"
                f"  Status : ❌ Cancelled"
            )
        if changed:
            _save_all(all_pos)
    except Exception as e:
        logger.error("[monitor] cancel_expired_pending error: %s", e)
    return cancelled


def _activate_pending(order_id: str, fill_price: float, notify_fn=None) -> bool:
    def _n(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass
    all_pos = _load_all()
    for i, pos in enumerate(all_pos):
        if pos.get("order_id") == order_id and pos.get("status") == "pending":
            all_pos[i]["status"]     = "open"
            all_pos[i]["fill_price"] = fill_price
            all_pos[i]["filled_at"]  = int(time.time())
            _save_all(all_pos)
            sym  = pos.get("symbol", "")
            side = pos.get("side", "BUY")
            emoji = "🟢" if side == "BUY" else "🔴"
            _n(
                f"{emoji} <b>LIMIT FILLED — {sym}</b>\n"
                f"  ID     : <code>{order_id}</code>\n"
                f"  Fill   : <code>{fill_price:.6f}</code>\n"
                f"  SL     : <code>{pos['stop_loss']}</code>  ✅ aktif\n"
                f"  TP     : <code>{pos['take_profit']}</code>  ✅ aktif\n"
                f"  Posisi live 🟢"
            )
            logger.info("[monitor] FILLED %s %s @ %.6f", order_id, sym, fill_price)
            return True
    logger.warning("[monitor] activate_pending: %s tidak ditemukan", order_id)
    return False


def _cancel_pending(order_id: str, reason: str, notify_fn=None) -> bool:
    def _n(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass
    all_pos = _load_all()
    for i, pos in enumerate(all_pos):
        if pos.get("order_id") == order_id and pos.get("status") == "pending":
            all_pos[i]["status"]        = "cancelled"
            all_pos[i]["cancel_reason"] = reason
            all_pos[i]["cancelled_at"]  = int(time.time())
            _save_all(all_pos)
            _n(f"❌ <b>CANCELLED — {pos.get('symbol')}</b> ID: <code>{order_id}</code> | {reason}")
            return True
    return False


def _count_session_filled(session_id: str) -> int:
    if not session_id:
        return 0
    return sum(
        1 for p in _load_all()
        if p.get("session_id") == session_id and p.get("status") == "open"
    )


def _cancel_session_pending(session_id: str, notify_fn=None) -> list:
    def _n(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass
    if not session_id:
        return []
    cancelled = []
    all_pos = _load_all()
    changed = False
    for i, pos in enumerate(all_pos):
        if pos.get("status") == "pending" and pos.get("session_id") == session_id:
            all_pos[i]["status"]        = "cancelled"
            all_pos[i]["cancel_reason"] = "sesi sudah 2 filled"
            all_pos[i]["cancelled_at"]  = int(time.time())
            changed = True
            sym = pos.get("symbol", "?")
            oid = pos.get("order_id", "?")
            cancelled.append(sym)
            _n(f"⚡ <b>Auto-Cancel — {sym}</b> ID: <code>{oid}</code> | sesi 2 filled")
            try:
                from order.executor import cancel_all_open_orders
                cancel_all_open_orders(sym)
            except Exception as e:
                logger.warning("[monitor] cancel_session Binance %s: %s", sym, e)
    if changed:
        _save_all(all_pos)
    return cancelled


# ------------------------------------------------------------------
# MonitoredPosition
# ------------------------------------------------------------------

@dataclass
class MonitoredPosition:
    raw:           dict
    symbol:        str
    side:          str
    entry_price:   float
    sl:            float
    tp:            float
    qty:           float
    notional:      float
    tick_size:     float
    risk:          float
    breakeven_hit: bool = False
    wti:           Optional[dict] = None
    opened_at:     float = field(default_factory=time.time)
    filled_at:     float = 0.0

    def calc_pnl(self, mark_price: float) -> float:
        if self.side == "BUY":
            return (mark_price - self.entry_price) / self.entry_price * self.notional
        return (self.entry_price - mark_price) / self.entry_price * self.notional

    def is_breakeven_triggered(self, mark_price: float) -> bool:
        pnl_pct  = self.calc_pnl(mark_price) / self.notional
        risk_pct = self.risk / self.entry_price
        return pnl_pct >= BREAKEVEN_RR * risk_pct

    def update_trailing_sl(self, mark_price: float) -> bool:
        if not self.breakeven_hit:
            return False
        tick = self.tick_size
        if self.side == "BUY":
            new_sl = round(round((mark_price - self.risk) / tick) * tick, 8)
            new_sl = max(new_sl, self.entry_price)
            if new_sl > self.sl + tick * 0.9:
                self.sl = new_sl
                return True
        else:
            new_sl = round(round((mark_price + self.risk) / tick) * tick, 8)
            new_sl = min(new_sl, self.entry_price)
            if new_sl < self.sl - tick * 0.9:
                self.sl = new_sl
                return True
        return False

    def is_sl_hit(self, price: float) -> bool:
        return price <= self.sl if self.side == "BUY" else price >= self.sl

    def is_tp_hit(self, price: float) -> bool:
        return price >= self.tp if self.side == "BUY" else price <= self.tp


# ------------------------------------------------------------------
# WsOrderManager — kirim order via Binance WebSocket API
# ------------------------------------------------------------------

class WsOrderManager:
    """
    Kirim market order / cancel semua order via Binance WebSocket API.
    Koneksi persistent → tidak ada TCP handshake, jauh lebih cepat dari REST.
    Dipakai untuk close posisi (SL/TP/CB) dan cancel pending saat CB.
    Jika WS putus → otomatis fallback ke REST executor.
    """

    def __init__(self, notify_fn: Optional[Callable] = None):
        self._notify    = notify_fn or (lambda msg, topic=None: None)
        self._ws        = None
        self._running   = False
        self._pending   : Dict[str, asyncio.Future] = {}
        self._lock      = asyncio.Lock()
        self._connected = asyncio.Event()

    async def start(self) -> None:
        self._running = True
        delay = RECONNECT_BASE
        while self._running:
            try:
                async with websockets.connect(
                    BINANCE_WS_API_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    delay = RECONNECT_BASE
                    logger.info("[ws_order] Connected")
                    async for raw in ws:
                        await self._on_message(raw)
            except asyncio.CancelledError:
                return
            except (ConnectionClosed, Exception) as e:
                self._ws = None
                self._connected.clear()
                logger.warning("[ws_order] Disconnected: %s — retry in %ds", e, delay)
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(ConnectionError("WS disconnected"))
                self._pending.clear()
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _on_message(self, raw: str) -> None:
        try:
            msg    = json.loads(raw)
            req_id = msg.get("id")
            if req_id and req_id in self._pending:
                fut = self._pending.pop(req_id)
                if not fut.done():
                    if msg.get("status") == 200:
                        fut.set_result(msg.get("result", {}))
                    else:
                        err = msg.get("error", {})
                        fut.set_exception(
                            Exception(f"WS API [{err.get('code')}]: {err.get('msg')}")
                        )
        except Exception as e:
            logger.warning("[ws_order] _on_message error: %s", e)

    async def _send(self, method: str, params: dict, timeout: float = 10.0) -> dict:
        """Kirim satu request ke WS API, tunggu response."""
        from config import BINANCE_API_KEY, BINANCE_API_SECRET, RECV_WINDOW
        await asyncio.wait_for(self._connected.wait(), timeout=5.0)

        req_id = str(uuid.uuid4())
        params["timestamp"]  = int(time.time() * 1000)
        params["recvWindow"] = RECV_WINDOW
        qs  = urllib.parse.urlencode(params)
        sig = hmac.new(BINANCE_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig

        payload = {
            "id":     req_id,
            "method": method,
            "params": {**params, "apiKey": BINANCE_API_KEY},
        }

        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        async with self._lock:
            self._pending[req_id] = fut
            await self._ws.send(json.dumps(payload))

        return await asyncio.wait_for(fut, timeout=timeout)

    async def close_position(self, symbol: str, side: str, qty: float) -> dict:
        """Close posisi via MARKET reduce-only. Fallback ke REST jika WS belum ready."""
        close_side = "SELL" if side == "BUY" else "BUY"
        try:
            result = await self._send("order.place", {
                "symbol":     symbol,
                "side":       close_side,
                "type":       "MARKET",
                "quantity":   str(qty),
                "reduceOnly": "true",
            })
            logger.info("[ws_order] close_position %s qty=%s id=%s",
                        symbol, qty, result.get("orderId"))
            return result
        except Exception as e:
            logger.error("[ws_order] close_position gagal %s: %s — fallback REST", symbol, e)
            loop = asyncio.get_event_loop()
            from order.executor import close_position_market
            return await loop.run_in_executor(
                None, lambda: close_position_market(symbol, side, qty)
            )

    async def cancel_all_open_orders(self, symbol: str) -> dict:
        """
        Cancel semua open order untuk symbol:
          1. Regular orders (LIMIT) via WS API openOrders.cancelAll
          2. Algo orders (SL/TP) via REST executor.cancel_all_open_orders
             — WS API tidak punya method cancel algo orders,
               executor sudah handle /fapi/v1/allOpenOrders + /fapi/v1/algoOpenOrders.

        Penting: sejak 2025-12-09 SL/TP adalah algo orders di /fapi/v1/algoOrder.
        Kalau hanya cancel regular orders, SL/TP Binance masih aktif dan bisa
        trigger lagi setelah posisi di-close manual → double close error.
        """
        loop = asyncio.get_event_loop()

        # Cancel regular orders via WS (cepat)
        result = {}
        try:
            result = await self._send("openOrders.cancelAll", {"symbol": symbol})
            logger.info("[ws_order] cancelAll regular orders %s", symbol)
        except Exception as e:
            logger.warning("[ws_order] cancelAll WS gagal %s: %s", symbol, e)

        # Cancel algo orders (SL/TP) via REST — tidak ada WS method untuk ini
        try:
            from order.executor import cancel_all_open_orders as _cancel_all_rest
            await loop.run_in_executor(None, lambda: _cancel_all_rest(symbol))
        except Exception as e:
            logger.error("[ws_order] cancelAll algo gagal %s: %s", symbol, e)

        return result


# ------------------------------------------------------------------
# UserDataStream — order fill/cancel real-time via WebSocket
# ------------------------------------------------------------------

class UserDataStream:
    """
    Subscribe ke Binance User Data Stream (listenKey).
    Terima ORDER_TRADE_UPDATE real-time → gantikan REST poll 5 detik.
    listenKey di-refresh tiap 30 menit agar tidak expire.
    """

    def __init__(self, on_event: Callable):
        self._on_event   = on_event
        self._listen_key = None
        self._running    = False

    async def start(self) -> None:
        self._running = True
        while self._running:
            try:
                self._listen_key = await self._get_listen_key()
                logger.info("[uds] listenKey: %s...", self._listen_key[:10])
                await asyncio.gather(
                    self._stream_loop(),
                    self._keepalive_loop(),
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("[uds] Error: %s — restart 5s", e)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        if self._listen_key:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._delete_listen_key, self._listen_key)

    async def _get_listen_key(self) -> str:
        from config import BINANCE_API_KEY
        loop = asyncio.get_event_loop()
        def _req():
            r = requests.post(
                f"{BINANCE_FUTURES_URL}/fapi/v1/listenKey",
                headers={"X-MBX-APIKEY": BINANCE_API_KEY},
                timeout=10,
            )
            r.raise_for_status()
            return r.json()["listenKey"]
        return await loop.run_in_executor(None, _req)

    def _delete_listen_key(self, listen_key: str) -> None:
        from config import BINANCE_API_KEY
        try:
            requests.delete(
                f"{BINANCE_FUTURES_URL}/fapi/v1/listenKey",
                params={"listenKey": listen_key},
                headers={"X-MBX-APIKEY": BINANCE_API_KEY},
                timeout=5,
            )
        except Exception:
            pass

    async def _keepalive_loop(self) -> None:
        from config import BINANCE_API_KEY
        while self._running:
            await asyncio.sleep(30 * 60)   # tiap 30 menit
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: requests.put(
                    f"{BINANCE_FUTURES_URL}/fapi/v1/listenKey",
                    params={"listenKey": self._listen_key},
                    headers={"X-MBX-APIKEY": BINANCE_API_KEY},
                    timeout=5,
                ))
                logger.debug("[uds] listenKey refreshed")
            except Exception as e:
                logger.warning("[uds] keepalive gagal: %s", e)

    async def _stream_loop(self) -> None:
        url   = f"wss://fstream.binance.com/ws/{self._listen_key}"
        delay = RECONNECT_BASE
        while self._running:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10, close_timeout=5
                ) as ws:
                    delay = RECONNECT_BASE
                    logger.info("[uds] User data stream connected")
                    async for raw in ws:
                        if not self._running:
                            return
                        try:
                            event = json.loads(raw)
                            await self._on_event(event)
                        except Exception as e:
                            logger.warning("[uds] Event error: %s", e)
            except asyncio.CancelledError:
                return
            except (ConnectionClosed, Exception) as e:
                logger.warning("[uds] Stream closed: %s — retry %ds", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX)
                # Refresh listenKey setelah disconnect panjang
                try:
                    self._listen_key = await self._get_listen_key()
                    url = f"wss://fstream.binance.com/ws/{self._listen_key}"
                except Exception:
                    pass


# ------------------------------------------------------------------
# PositionMonitor
# ------------------------------------------------------------------

class PositionMonitor:

    def __init__(
        self,
        notify: Optional[Callable[[str], None]] = None,
        poll_interval: float = 5.0,
    ):
        self._notify_fn    = notify or (lambda msg: None)
        self.poll_interval = poll_interval

        self.positions:   Dict[str, MonitoredPosition] = {}
        self.vol_analyzer = VolumeAnalyzer(spike_multiplier=6.5)

        self._ws_tasks:   Dict[str, asyncio.Task] = {}
        self._running     = False
        self._last_price:  Dict[str, float] = {}
        self._last_bucket: Dict[str, float] = {}
        self._filling:     Set[str] = set()

        self._btc_prev_close:   float = 0.0
        self._btc_last_price:   float = 0.0
        self._urgent_cb_bucket: float = 0.0
        self._btc_close_date:   str   = ""

        # WebSocket API — order execution (close/cancel) + User Data Stream
        self._ws_order_mgr = WsOrderManager(notify_fn=self.notify)
        self._user_data_ws = UserDataStream(on_event=self._on_user_data_event)

    def notify(self, msg: str, topic: int = TOPIC_ORDERS) -> None:
        try:
            self._notify_fn(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._btc_prev_close = await asyncio.get_event_loop().run_in_executor(
            None, fetch_btc_prev_close
        )
        logger.info("[monitor] BTC prev close: %.2f", self._btc_prev_close)

        # Launch WebSocket API manager + User Data Stream
        asyncio.create_task(self._ws_order_mgr.start(), name="ws-order-mgr")
        asyncio.create_task(self._user_data_ws.start(),  name="ws-user-data")

        await self._sync_positions()
        await self._notify_startup()
        asyncio.create_task(self._sync_loop(), name="monitor-sync")

    async def stop(self) -> None:
        self._running = False
        await self._ws_order_mgr.stop()
        await self._user_data_ws.stop()
        for task in self._ws_tasks.values():
            task.cancel()
        self._ws_tasks.clear()

    # ------------------------------------------------------------------
    # Sync loop — fallback reconciliation tiap poll_interval
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.poll_interval)
            await self._sync_positions()

    async def _sync_positions(self) -> None:
        cancel_expired_pending(notify_fn=self.notify)
        # _check_pending_fills tetap jalan sebagai fallback jika User Data Stream miss event
        await self._check_pending_fills()

        open_positions  = load_open_positions()
        open_symbols    = {p.get("symbol", "").upper() for p in open_positions}
        current_symbols = set(self.positions.keys())

        for pos_data in open_positions:
            sym = pos_data.get("symbol", "").upper()
            if sym and sym not in self.positions:
                await self._init_position(sym, pos_data)

        for sym in current_symbols - open_symbols:
            self._cleanup_position(sym)

        self._ensure_pending_streams()

        # Cek stale momentum cut untuk semua posisi open
        for sym, pos in list(self.positions.items()):
            asyncio.create_task(self._check_stale_momentum_cut(pos))

        if self.positions:
            self._flush_sl_to_file()

    # ------------------------------------------------------------------
    # User Data Stream handler — real-time order fill/cancel
    # ------------------------------------------------------------------

    async def _on_user_data_event(self, event: dict) -> None:
        """
        Handler User Data Stream events:
          - ORDER_TRADE_UPDATE  → entry fill, SL/TP fill (cp=true), cancel
          - CONDITIONAL_ORDER_TRIGGER_REJECT → SL/TP gagal trigger → fallback close
        """
        evt_type = event.get("e")
        loop     = asyncio.get_event_loop()

        # ── CONDITIONAL_ORDER_TRIGGER_REJECT ──────────────────────────────
        # SL atau TP order sudah triggered tapi Binance tolak eksekusinya.
        # Harus fallback close manual via market order.
        if evt_type == "CONDITIONAL_ORDER_TRIGGER_REJECT":
            or_    = event.get("or", {})
            symbol = or_.get("s", "").upper()
            oid    = int(or_.get("i", 0))
            reason = or_.get("r", "unknown")
            logger.error("[monitor] CONDITIONAL_ORDER_TRIGGER_REJECT %s id=%s: %s",
                         symbol, oid, reason)
            self.notify(
                f"🚨 <b>SL/TP REJECTED — {symbol}</b>\n"
                f"  ID     : <code>{oid}</code>\n"
                f"  Reason : <code>{reason}</code>\n"
                f"  ⚠️ Fallback market close...",
                TOPIC_ERROR,
            )
            # Cari posisi yang punya sl_order_id atau tp_order_id ini
            pos = self.positions.get(symbol)
            if pos:
                mark = self._last_price.get(symbol) or get_mark_price(symbol)
                if mark:
                    await self._close_position(pos, mark, reason=f"SL/TP rejected fallback 🚨")
            return

        # ── ORDER_TRADE_UPDATE ────────────────────────────────────────────
        if evt_type != "ORDER_TRADE_UPDATE":
            return

        order      = event.get("o", {})
        symbol     = order.get("s", "").upper()
        binance_id = int(order.get("i", 0))
        status     = order.get("X", "")        # NEW, FILLED, CANCELED, EXPIRED, EXPIRED_IN_MATCH
        avg_px     = float(order.get("ap") or 0)
        close_all  = bool(order.get("cp", False))   # true = SL/TP closePosition order
        order_type = order.get("o", "")             # STOP_MARKET, TAKE_PROFIT_MARKET, LIMIT, dll

        logger.info("[monitor] ORDER_TRADE_UPDATE %s id=%s status=%s type=%s cp=%s px=%.6f",
                    symbol, binance_id, status, order_type, close_all, avg_px)

        if not binance_id:
            return

        # ── Case 1: SL atau TP Binance terfill (closePosition=true) ──────
        # Binance kirim cp=true saat conditional order (SL/TP) berhasil eksekusi.
        # Posisi sudah ditutup oleh Binance — kita hanya perlu update JSON + notif.
        if close_all and status in ("FILLED", "EXPIRED_IN_MATCH"):
            pos = self.positions.get(symbol)
            if pos:
                fill_px = avg_px or self._last_price.get(symbol) or get_mark_price(symbol) or 0.0
                # Tentukan apakah ini SL atau TP berdasarkan order_type
                is_sl = order_type in ("STOP_MARKET", "STOP")
                is_tp = order_type in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT")

                if is_sl:
                    reason = "Trailing SL 🔄" if pos.breakeven_hit else "SL Binance ✅"
                elif is_tp:
                    reason = "TP Binance ✅"
                else:
                    reason = f"Conditional {order_type} filled"

                logger.info("[monitor] SL/TP filled by Binance %s @ %.6f reason=%s",
                            symbol, fill_px, reason)

                # Posisi sudah ditutup Binance — jangan kirim market order lagi,
                # langsung update JSON saja
                await self._record_binance_close(pos, fill_px, reason)
            return

        # ── Case 2: Entry order fill (pending → open) ─────────────────────
        if status == "FILLED" and not close_all:
            pending = load_pending_positions()
            for pos in pending:
                if int(pos.get("binance_order_id", 0)) == binance_id:
                    order_id   = pos.get("order_id", "")
                    session_id = pos.get("session_id", "")

                    if order_id in self._filling:
                        return
                    self._filling.add(order_id)
                    try:
                        fill_price = avg_px or float(pos.get("entry_price", 0))
                        await loop.run_in_executor(
                            None,
                            lambda: _activate_pending(order_id, fill_price, notify_fn=self.notify),
                        )
                    finally:
                        self._filling.discard(order_id)

                    if session_id:
                        total = _count_session_filled(session_id)
                        if total >= MAX_FILLS_PER_SESSION:
                            await loop.run_in_executor(
                                None,
                                lambda: _cancel_session_pending(session_id, notify_fn=self.notify),
                            )

                    await self._sync_positions()
                    break

        # ── Case 3: Entry order cancel/expire ─────────────────────────────
        elif status in ("CANCELED", "EXPIRED", "EXPIRED_IN_MATCH") and not close_all:
            pending = load_pending_positions()
            for pos in pending:
                if int(pos.get("binance_order_id", 0)) == binance_id:
                    order_id = pos.get("order_id", "")
                    await loop.run_in_executor(
                        None,
                        lambda: _cancel_pending(
                            order_id,
                            f"Binance status: {status}",
                            notify_fn=self.notify,
                        ),
                    )
                    break

    async def _record_binance_close(self, pos: MonitoredPosition, price: float,
                                     reason: str) -> None:
        """
        Update JSON + notif saat SL/TP sudah dieksekusi langsung oleh Binance.
        Tidak kirim market order (posisi sudah closed di Binance).
        """
        symbol = pos.symbol
        if symbol not in self.positions:
            return

        pnl      = round(pos.calc_pnl(price) - pos.notional * TAKER_FEE * 2, 4)
        hold_min = (time.time() - pos.opened_at) / 60

        if "TP" in reason:
            close_status = "TP"
            should_ban   = False
        else:
            close_status = "SL"
            should_ban   = True

        raw = pos.raw.copy()
        raw.update({
            "status":       close_status,
            "close_reason": reason,
            "close_price":  price,
            "pnl":          pnl,
            "closed_at":    time.time(),
            "hold_minutes": round(hold_min, 1),
            "sl_final":     pos.sl,
        })
        all_pos = _load_all()
        for i, pd in enumerate(all_pos):
            if pd.get("order_id") == raw.get("order_id"):
                all_pos[i] = raw
                break
        _save_all(all_pos)
        _append_history(raw)

        if should_ban:
            try:
                register_sl(symbol)
            except Exception as e:
                logger.warning("[monitor] register_sl error: %s", e)

        self._cleanup_position(symbol)

        pnl_str  = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        ban_note = "\n  🚫 Simbol di-ban 2 sesi" if should_ban else ""
        emoji    = "✅" if close_status == "TP" else "🛑"

        self.notify(
            f"{emoji} <b>CLOSED (Binance) — {symbol}</b>\n"
            f"  Reason : <i>{reason}</i>\n"
            f"  Entry  : <code>{pos.entry_price}</code>\n"
            f"  Close  : <code>{price:.6f}</code>\n"
            f"  SL     : <code>{pos.sl:.6f}</code>\n"
            f"  PnL    : <b>{pnl_str} USDT</b>\n"
            f"  Hold   : {hold_min:.0f} menit"
            f"{ban_note}",
            TOPIC_ORDERS,
        )
        logger.info("[monitor] CLOSED (Binance SL/TP) %s @ %.6f status=%s pnl=%.4f",
                    symbol, price, close_status, pnl)

    # ------------------------------------------------------------------
    # Pending fill — REST fallback (tetap jalan sebagai safety net)
    # ------------------------------------------------------------------

    async def _check_pending_fills(self) -> None:
        pending = load_pending_positions()
        if not pending:
            return

        loop = asyncio.get_event_loop()

        for pos in pending:
            sym         = pos.get("symbol", "").upper()
            order_id    = pos.get("order_id", "")
            binance_oid = pos.get("binance_order_id")
            session_id  = pos.get("session_id", "")

            if not order_id or not binance_oid:
                continue
            if order_id in self._filling:
                continue

            try:
                from order.executor import get_order_status
                info   = await loop.run_in_executor(
                    None, get_order_status, sym, int(binance_oid)
                )
                status = info.get("status", "")

                if status == "FILLED":
                    fill_price = float(info.get("avgPrice", 0)) or float(pos.get("entry_price", 0))
                    logger.info("[monitor] REST fallback FILLED %s %s @ %.6f", order_id, sym, fill_price)

                    self._filling.add(order_id)
                    try:
                        await loop.run_in_executor(
                            None,
                            lambda: _activate_pending(order_id, fill_price, notify_fn=self.notify),
                        )
                    finally:
                        self._filling.discard(order_id)

                    if session_id:
                        total = _count_session_filled(session_id)
                        if total >= MAX_FILLS_PER_SESSION:
                            await loop.run_in_executor(
                                None,
                                lambda: _cancel_session_pending(session_id, notify_fn=self.notify),
                            )

                elif status in ("CANCELED", "EXPIRED", "REJECTED"):
                    await loop.run_in_executor(
                        None,
                        lambda: _cancel_pending(
                            order_id, f"Binance status: {status}", notify_fn=self.notify
                        ),
                    )

            except Exception as e:
                logger.warning("[monitor] REST fallback cek order %s %s: %s", sym, binance_oid, e)

    # ------------------------------------------------------------------
    # WS stream untuk pending
    # ------------------------------------------------------------------

    def _ensure_pending_streams(self) -> None:
        for pos in load_pending_positions():
            sym = pos.get("symbol", "").upper()
            if sym:
                self._start_stream(sym)
                self._ensure_btc_stream()

    # ------------------------------------------------------------------
    # Init / cleanup posisi
    # ------------------------------------------------------------------

    async def _init_position(self, symbol: str, data: dict) -> None:
        tick  = get_tick_size(symbol)
        entry = float(data.get("entry_price", 0))
        sl    = float(data.get("stop_loss", 0))
        tp    = float(data.get("take_profit", 0))
        side  = data.get("side", "BUY").upper()

        sl_initial    = float(data.get("sl_initial", sl))
        risk          = abs(entry - sl_initial)
        breakeven_hit = bool(data.get("breakeven_hit", False))

        pos = MonitoredPosition(
            raw           = data,
            symbol        = symbol,
            side          = side,
            entry_price   = entry,
            sl            = sl,
            tp            = tp,
            qty           = float(data.get("qty", 0)),
            notional      = float(data.get("notional", 0)),
            tick_size     = tick,
            risk          = risk,
            breakeven_hit = breakeven_hit,
            opened_at     = float(data.get("opened_at", time.time())),
            filled_at     = float(data.get("filled_at", 0)),
        )

        loop = asyncio.get_event_loop()
        wti  = await loop.run_in_executor(None, get_wti, symbol)
        pos.wti = wti

        # Simpan qty_original jika belum ada — dipakai oleh stale_momentum_cut
        if not data.get("qty_original"):
            pos.raw["qty_original"] = float(data.get("qty", 0))
            try:
                all_pos = _load_all()
                for i, pd in enumerate(all_pos):
                    if pd.get("order_id") == data.get("order_id"):
                        all_pos[i]["qty_original"] = pos.raw["qty_original"]
                        break
                _save_all(all_pos)
            except Exception as e:
                logger.warning("[monitor] simpan qty_original gagal %s: %s", symbol, e)

        self.positions[symbol] = pos
        self.vol_analyzer.init_symbol(symbol)
        self._last_bucket[symbol] = time.time()

        self._start_stream(symbol)
        self._ensure_btc_stream()

        wti_tag = ""
        if wti:
            wti_tag = (
                f"\n  WTI vs BTC : <b>{wti['wti_pct']:.1f}%</b> "
                f"{'✅ BTC reversal aktif' if wti['btc_active'] else '⚪ BTC diabaikan'}"
            )
        be_tag = "\n  ⚖️ <b>Breakeven aktif</b> — trailing SL restored ✅" if breakeven_hit else ""

        self.notify(
            f"👁 <b>Monitor aktif — {symbol}</b>\n"
            f"  Side  : <b>{side}</b>\n"
            f"  Entry : <code>{entry}</code>\n"
            f"  SL    : <code>{sl}</code>{'  (trailing)' if breakeven_hit else ''}\n"
            f"  TP    : <code>{tp}</code>\n"
            f"  Risk  : <code>{risk:.6f}</code>"
            f"{be_tag}{wti_tag}",
            TOPIC_ORDERS,
        )
        logger.info("[monitor] Init %s side=%s entry=%.6f sl=%.6f be=%s WTI=%s",
                    symbol, side, entry, sl, breakeven_hit,
                    wti["wti_pct"] if wti else "N/A")

    def _cleanup_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)
        self.vol_analyzer.remove_symbol(symbol)
        self._last_bucket.pop(symbol, None)
        self._last_price.pop(symbol, None)
        task = self._ws_tasks.pop(symbol, None)
        if task:
            task.cancel()
        if not self.positions and not load_pending_positions():
            btc = self._ws_tasks.pop("BTCUSDT", None)
            if btc:
                btc.cancel()
        logger.info("[monitor] Cleanup %s", symbol)

    # ------------------------------------------------------------------
    # SL flush ke file
    # ------------------------------------------------------------------

    def _flush_sl_to_file(self) -> None:
        try:
            all_pos = _load_all()
            changed = False
            for i, pd in enumerate(all_pos):
                sym = pd.get("symbol", "").upper()
                mp  = self.positions.get(sym)
                if not mp or pd.get("status") != "open":
                    continue
                saved_sl = float(pd.get("stop_loss", 0))
                if abs(mp.sl - saved_sl) > 1e-10 or pd.get("breakeven_hit") != mp.breakeven_hit:
                    all_pos[i]["stop_loss"]     = mp.sl
                    all_pos[i]["breakeven_hit"] = mp.breakeven_hit
                    if mp.raw.get("sl_order_id"):
                        all_pos[i]["sl_order_id"] = mp.raw["sl_order_id"]
                    changed = True
            if changed:
                _save_all(all_pos)
        except Exception as e:
            logger.error("[monitor] flush_sl error: %s", e)

    # ------------------------------------------------------------------
    # Close posisi — via WsOrderManager (WebSocket API) + REST fallback
    # ------------------------------------------------------------------

    async def _close_position(self, pos: MonitoredPosition, price: float,
                               reason: str = "") -> None:
        symbol = pos.symbol
        if symbol not in self.positions:
            return

        pnl      = round(pos.calc_pnl(price) - pos.notional * TAKER_FEE * 2, 4)
        hold_min = (time.time() - pos.opened_at) / 60

        if "Max Hold" in reason:
            close_status = "TP" if pnl >= 0 else "SL"
            should_ban   = False
        elif "TP tercapai" in reason or "Trailing SL" in reason:
            close_status = "TP"
            should_ban   = False
        else:
            close_status = "SL"
            should_ban   = True

        try:
            # Cancel semua open order (SL/TP Binance) via WS dulu
            await self._ws_order_mgr.cancel_all_open_orders(symbol)
            # Close posisi via WS API (persistent connection, lebih cepat dari REST)
            await self._ws_order_mgr.close_position(symbol, pos.side, pos.qty)
            logger.info("[monitor] Close WS %s qty=%.6f @ %.6f", symbol, pos.qty, price)
        except Exception as e:
            logger.error("[monitor] Gagal close %s: %s", symbol, e)
            self.notify(
                f"🚨 <b>Close GAGAL — {symbol}</b>\n"
                f"  <code>{e}</code>\n"
                f"  ⚠️ Cek posisi manual!",
                TOPIC_ERROR,
            )
            return

        # Update JSON
        raw = pos.raw.copy()
        raw.update({
            "status":       close_status,
            "close_reason": reason,
            "close_price":  price,
            "pnl":          pnl,
            "closed_at":    time.time(),
            "hold_minutes": round(hold_min, 1),
            "sl_final":     pos.sl,
        })
        all_pos = _load_all()
        for i, pd in enumerate(all_pos):
            if pd.get("order_id") == raw.get("order_id"):
                all_pos[i] = raw
                break
        _save_all(all_pos)
        _append_history(raw)

        if should_ban:
            try:
                register_sl(symbol)
            except Exception as e:
                logger.warning("[monitor] register_sl error: %s", e)

        self._cleanup_position(symbol)

        pnl_str  = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        ban_note = "\n  🚫 Simbol di-ban 2 sesi" if should_ban else ""
        if "Max Hold" in reason:
            emoji = "⏰✅" if pnl >= 0 else "⏰🛑"
        else:
            emoji = "✅" if close_status == "TP" else "🛑"

        self.notify(
            f"{emoji} <b>CLOSED — {symbol}</b>\n"
            f"  Reason : <i>{reason}</i>\n"
            f"  Entry  : <code>{pos.entry_price}</code>\n"
            f"  Close  : <code>{price:.6f}</code>\n"
            f"  SL     : <code>{pos.sl:.6f}</code>\n"
            f"  PnL    : <b>{pnl_str} USDT</b>\n"
            f"  Hold   : {hold_min:.0f} menit"
            f"{ban_note}",
            TOPIC_ORDERS,
        )
        logger.info("[monitor] CLOSED %s @ %.6f status=%s pnl=%.4f",
                    symbol, price, close_status, pnl)

    # ------------------------------------------------------------------
    # WebSocket streams (aggTrade)
    # ------------------------------------------------------------------

    def _start_stream(self, symbol: str) -> None:
        sym = symbol.upper()
        existing = self._ws_tasks.get(sym)
        if existing and not existing.done():
            return
        self._ws_tasks[sym] = asyncio.create_task(
            self._stream_agg_trade(sym), name=f"monitor-ws-{sym}"
        )

    def _ensure_btc_stream(self) -> None:
        existing = self._ws_tasks.get("BTCUSDT")
        if existing and not existing.done():
            return
        self.vol_analyzer.init_symbol("BTCUSDT")
        self._ws_tasks["BTCUSDT"] = asyncio.create_task(
            self._stream_agg_trade("BTCUSDT"), name="monitor-ws-BTCUSDT"
        )

    async def _stream_agg_trade(self, symbol: str) -> None:
        url   = f"{BINANCE_WS_BASE}?streams={symbol.lower()}@aggTrade"
        delay = RECONNECT_BASE

        while self._running:
            has_open    = symbol in self.positions
            has_pending = any(
                p.get("symbol", "").upper() == symbol for p in load_pending_positions()
            )
            if symbol != "BTCUSDT" and not has_open and not has_pending:
                logger.info("[monitor] WS %s stop — tidak ada posisi/pending", symbol)
                return
            if symbol == "BTCUSDT" and not self.positions and not load_pending_positions():
                logger.info("[monitor] WS BTC stop")
                return

            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10, close_timeout=5
                ) as ws:
                    delay = RECONNECT_BASE
                    logger.info("[monitor] WS connected: %s", symbol)
                    async for raw in ws:
                        if not self._running:
                            return
                        try:
                            msg  = json.loads(raw)
                            data = msg.get("data", msg)
                            await self._on_agg_trade(symbol, data)
                        except Exception as e:
                            logger.warning("[monitor] Dispatch %s: %s", symbol, e)
            except asyncio.CancelledError:
                return
            except ConnectionClosed as e:
                logger.warning("[monitor] WS closed %s: %s", symbol, e)
            except Exception as e:
                logger.warning("[monitor] WS error %s: %s", symbol, e)

            if not self._running:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX)

    # ------------------------------------------------------------------
    # Core: proses setiap aggTrade
    # ------------------------------------------------------------------

    async def _on_agg_trade(self, symbol: str, data: dict) -> None:
        price          = float(data.get("p", 0))
        qty            = float(data.get("q", 0))
        is_buyer_maker = bool(data.get("m", False))

        if price <= 0 or qty <= 0:
            return

        self._last_price[symbol] = price
        self.vol_analyzer.feed(symbol, price, qty, is_buyer_maker)

        if symbol == "BTCUSDT":
            self._btc_last_price = price

        # Evaluate posisi open
        if symbol != "BTCUSDT":
            pos = self.positions.get(symbol)
            if pos:
                await self._evaluate_position(pos, price)

        # Bucket checks
        now = time.time()
        if symbol != "BTCUSDT":
            if now - self._last_bucket.get(symbol, 0) >= 60:
                self._last_bucket[symbol] = now
                pos = self.positions.get(symbol)
                if pos:
                    await self._check_volume_reversal(pos)
        else:
            if now - self._last_bucket.get("BTCUSDT", 0) >= 60:
                self._last_bucket["BTCUSDT"] = now
                for sym, pos in list(self.positions.items()):
                    if pos.wti and pos.wti.get("btc_active"):
                        await self._check_btc_volume_reversal(pos)

            if now - self._urgent_cb_bucket >= URGENT_CB_BUCKET_SEC:
                self._urgent_cb_bucket = now
                await self._check_urgent_cb()

    # ------------------------------------------------------------------
    # Evaluate posisi per tick
    # ------------------------------------------------------------------

    async def _evaluate_position(self, pos: MonitoredPosition, price: float) -> None:
        symbol = pos.symbol

        ref_ts = pos.filled_at if pos.filled_at > 0 else pos.opened_at
        if time.time() - ref_ts >= MAX_HOLD_SEC:
            await self._close_position(
                pos, price, reason=f"Max Hold {(time.time()-ref_ts)/3600:.1f}h ⏰"
            )
            return

        if pos.is_tp_hit(price):
            await self._close_position(pos, price, reason="TP tercapai ✅")
            return

        if pos.is_sl_hit(price):
            reason = "Trailing SL 🔄" if pos.breakeven_hit else "SL tercapai 🛑"
            await self._close_position(pos, price, reason=reason)
            return

        await self._check_partial_tp(pos, price)

        if not pos.breakeven_hit and pos.is_breakeven_triggered(price):
            pos.breakeven_hit = True
            pos.sl = pos.entry_price
            self._flush_sl_to_file()

            loop = asyncio.get_event_loop()
            try:
                from order.executor import amend_sl_to_price
                sl_oid = pos.raw.get("sl_order_id")
                if sl_oid:
                    tick     = pos.tick_size
                    be_price = round(round(pos.entry_price / tick) * tick, 8)
                    new_sl_id = await loop.run_in_executor(
                        None,
                        lambda: amend_sl_to_price(symbol, sl_oid, pos.side, be_price),
                    )
                    if new_sl_id:
                        pos.raw["sl_order_id"]  = new_sl_id
                        pos.raw["breakeven_hit"] = True
                        self._flush_sl_to_file()
                        logger.info("[monitor] BE SL amended %s @ %.6f id=%s",
                                    symbol, be_price, new_sl_id)
            except Exception as e:
                logger.error("[monitor] amend_breakeven_sl error %s: %s", symbol, e)

            self.notify(
                f"⚖️ <b>Breakeven — {symbol}</b>\n"
                f"  SL digeser ke entry: <code>{pos.entry_price}</code>\n"
                f"  SL Binance diupdate ✅\n"
                f"  Trailing stop aktif ✅",
                TOPIC_ORDERS,
            )

        if pos.breakeven_hit:
            if pos.update_trailing_sl(price):
                self._flush_sl_to_file()

    # ------------------------------------------------------------------
    # Partial TP @ RR 1.5 — 30% close via WsOrderManager
    # ------------------------------------------------------------------

    async def _check_partial_tp(self, pos: MonitoredPosition, price: float) -> None:
        if pos.raw.get("partial_tp_done"):
            return
        entry = pos.entry_price
        sl    = float(pos.raw.get("sl_initial", pos.sl))
        risk  = abs(entry - sl)
        if risk <= 0:
            return
        hit = (pos.side == "BUY" and price >= entry + PARTIAL_TP_RR * risk) or \
              (pos.side == "SELL" and price <= entry - PARTIAL_TP_RR * risk)
        if not hit:
            return

        step        = get_qty_step(pos.symbol)
        raw_qty     = pos.qty * PARTIAL_TP_PCT
        partial_qty = floor_qty(raw_qty, step)
        if partial_qty <= 0:
            logger.warning("[monitor] Partial TP %s qty terlalu kecil setelah floor (%.6f step=%.6f)",
                           pos.symbol, raw_qty, step)
            return
        partial_notional = pos.notional * PARTIAL_TP_PCT
        if pos.side == "BUY":
            partial_pnl = (price - entry) / entry * partial_notional
        else:
            partial_pnl = (entry - price) / entry * partial_notional
        partial_pnl = round(partial_pnl - partial_notional * TAKER_FEE * 2, 4)

        try:
            # Partial close via WS API
            await self._ws_order_mgr.close_position(pos.symbol, pos.side, partial_qty)
        except Exception as e:
            logger.error("[monitor] Partial TP close gagal %s: %s", pos.symbol, e)
            self.notify(f"🚨 <b>{pos.symbol}</b> — Partial TP gagal: <code>{e}</code>", TOPIC_ERROR)
            return

        remaining_qty    = floor_qty(pos.qty - partial_qty, step)
        pos.qty      = remaining_qty
        pos.notional = round(pos.notional * (1 - PARTIAL_TP_PCT), 4)
        pos.raw["qty"]             = pos.qty
        pos.raw["notional"]        = pos.notional
        pos.raw["partial_tp_done"] = True
        pos.raw["partial_tp_pnl"]  = partial_pnl
        self._flush_sl_to_file()

        pnl_str = f"+{partial_pnl:.2f}" if partial_pnl >= 0 else f"{partial_pnl:.2f}"
        self.notify(
            f"🎯 <b>Partial TP 30% — {pos.symbol}</b>\n"
            f"  Price  : <code>{price:.6f}</code>  (RR {PARTIAL_TP_RR}:1)\n"
            f"  PnL    : <b>{pnl_str} USDT</b>\n"
            f"  Sisa   : 70% posisi masih aktif ✅",
            TOPIC_ORDERS,
        )
        logger.info("[monitor] Partial TP %s qty=%.6f pnl=%.4f", pos.symbol, partial_qty, partial_pnl)

    # ------------------------------------------------------------------
    # Stale Momentum Cut — partial 40% jika posisi stagnan >12 jam
    # dan 3 candle 4H terakhir bergerak kuat melawan posisi
    # ------------------------------------------------------------------

    async def _check_stale_momentum_cut(self, pos: MonitoredPosition) -> None:
        """
        Safety valve: partial cut 40% dari qty_original jika:
          1. Posisi sudah ≥ 12 jam
          2. Flag stale_momentum_cut_done belum di-set
          3. Close candle 4H terlama → terbaru (dari 3 closed) melawan posisi
          4. Magnitude pergerakan ≥ max(1%, 4 × ATR-14)

        Setelah cut:
          - SL lama di-cancel → dipasang ulang dengan qty sisa (60% dari original)
          - JSON diupdate, notif dikirim ke Telegram
        """
        symbol = pos.symbol

        # ── Guard 1: umur posisi ─────────────────────────────────────────────
        ref_ts = pos.filled_at if pos.filled_at > 0 else pos.opened_at
        age    = time.time() - ref_ts
        if age < STALE_MOMENTUM_MIN_AGE_SEC:
            return

        # ── Guard 2: flag belum done ─────────────────────────────────────────
        if pos.raw.get("stale_momentum_cut_done"):
            return

        # ── Fetch klines 4H ──────────────────────────────────────────────────
        # Butuh 14+1 candle untuk ATR-14 + minimal 3 candle closed terakhir.
        # Fetch 20: kline[-1] = sedang berjalan (skip), [-4:-1] = 3 closed.
        loop   = asyncio.get_event_loop()
        klines = await loop.run_in_executor(None, _fetch_4h_klines, symbol, 20)
        if len(klines) < STALE_MOMENTUM_ATR_PERIOD + 4:
            logger.warning("[monitor] stale_momentum %s: klines tidak cukup (%d)",
                           symbol, len(klines))
            return

        # Buang candle yang sedang berjalan (belum closed)
        closed_candles = klines[:-1]
        c1 = closed_candles[-3]   # terlama dari 3 candle closed terbaru
        c3 = closed_candles[-1]   # terbaru dari 3 candle closed terbaru

        close_c1 = float(c1[4])
        close_c3 = float(c3[4])

        # ── Guard 3: arah melawan posisi ─────────────────────────────────────
        # SHORT melawan = harga naik  (close_c3 > close_c1)
        # LONG  melawan = harga turun (close_c3 < close_c1)
        if pos.side == "SELL":
            if close_c3 <= close_c1:
                return
        else:
            if close_c3 >= close_c1:
                return

        # ── Guard 4: magnitude ≥ max(1%, 4×ATR) ─────────────────────────────
        move_abs = abs(close_c3 - close_c1)
        move_pct = move_abs / close_c1 if close_c1 > 0 else 0.0

        atr            = _calc_atr(closed_candles, STALE_MOMENTUM_ATR_PERIOD)
        atr_threshold  = STALE_MOMENTUM_ATR_MULT * atr if atr > 0 else 0.0
        threshold_abs  = max(STALE_MOMENTUM_MIN_MOVE_PCT * close_c1, atr_threshold)

        if move_abs < threshold_abs:
            logger.info(
                "[monitor] stale_momentum %s: move=%.6f threshold=%.6f "
                "(1%%=%.6f 4×ATR=%.6f) — skip",
                symbol, move_abs, threshold_abs,
                STALE_MOMENTUM_MIN_MOVE_PCT * close_c1, atr_threshold,
            )
            return

        # ════════════════════════════════════════════════════════════════════
        # TRIGGER — eksekusi partial cut
        # ════════════════════════════════════════════════════════════════════
        logger.info(
            "[monitor] stale_momentum TRIGGER %s side=%s age=%.1fh "
            "close_c1=%.6f close_c3=%.6f move=%.2f%% threshold=%.2f%% ATR=%.6f",
            symbol, pos.side, age / 3600,
            close_c1, close_c3, move_pct * 100,
            threshold_abs / close_c1 * 100 if close_c1 > 0 else 0, atr,
        )

        # ── Hitung qty ────────────────────────────────────────────────────────
        qty_original = float(pos.raw.get("qty_original", pos.qty))
        step         = get_qty_step(symbol)
        cut_qty      = floor_qty(qty_original * STALE_MOMENTUM_CUT_PCT, step)
        sisa_qty     = floor_qty(qty_original * (1.0 - STALE_MOMENTUM_CUT_PCT), step)

        if cut_qty <= 0 or sisa_qty <= 0:
            logger.warning("[monitor] stale_momentum %s: qty terlalu kecil cut=%.6f sisa=%.6f",
                           symbol, cut_qty, sisa_qty)
            return

        mark = self._last_price.get(symbol) or get_mark_price(symbol) or 0.0

        # ── Step 1: Partial close 40% via WS market reduce-only ──────────────
        try:
            await self._ws_order_mgr.close_position(symbol, pos.side, cut_qty)
            logger.info("[monitor] stale_momentum partial close %s qty=%.6f", symbol, cut_qty)
        except Exception as e:
            logger.error("[monitor] stale_momentum partial close GAGAL %s: %s", symbol, e)
            self.notify(
                f"🚨 <b>{symbol}</b> — Stale Momentum Cut GAGAL partial close:\n"
                f"  <code>{e}</code>",
                TOPIC_ERROR,
            )
            return

        # ── Step 2: Cancel SL lama → pasang SL baru dengan qty sisa ─────────
        old_sl_algo_id = pos.raw.get("sl_order_id")
        sl_price       = float(pos.raw.get("stop_loss", pos.sl))
        sl_side        = "SELL" if pos.side == "BUY" else "BUY"
        new_sl_id      = None

        if old_sl_algo_id:
            try:
                from order.executor import cancel_algo_order
                await loop.run_in_executor(
                    None, cancel_algo_order, symbol, int(old_sl_algo_id)
                )
                logger.info("[monitor] stale_momentum cancel SL lama %s algoId=%s",
                            symbol, old_sl_algo_id)
            except Exception as e:
                logger.warning("[monitor] stale_momentum cancel SL lama gagal %s: %s", symbol, e)

            try:
                from order.executor import place_stop_market
                resp      = await loop.run_in_executor(
                    None,
                    lambda: place_stop_market(symbol, sl_side, sl_price, qty=sisa_qty),
                )
                new_sl_id = resp.get("algoId")
                logger.info("[monitor] stale_momentum SL baru %s qty=%.6f @ %.6f algoId=%s",
                            symbol, sisa_qty, sl_price, new_sl_id)
            except Exception as e:
                logger.error("[monitor] stale_momentum pasang SL baru GAGAL %s: %s", symbol, e)
                self.notify(
                    f"🚨 <b>{symbol}</b> — Stale Momentum: SL baru GAGAL dipasang!\n"
                    f"  Qty sisa <code>{sisa_qty}</code> tanpa SL — cek manual!",
                    TOPIC_ERROR,
                )
        else:
            logger.warning("[monitor] stale_momentum %s: tidak ada sl_order_id di raw", symbol)

        # ── Step 3: Update MonitoredPosition in-memory ───────────────────────
        pos.qty            = sisa_qty
        pos.raw["qty"]                     = sisa_qty
        pos.raw["stale_momentum_cut_done"] = True
        pos.raw["stale_momentum_cut_at"]   = int(time.time())
        pos.raw["stale_momentum_cut_qty"]  = cut_qty
        if new_sl_id:
            pos.raw["sl_order_id"] = new_sl_id

        # ── Step 4: Flush ke JSON ─────────────────────────────────────────────
        try:
            all_pos  = _load_all()
            order_id = pos.raw.get("order_id")
            for i, pd in enumerate(all_pos):
                if pd.get("order_id") == order_id:
                    all_pos[i]["qty"]                     = sisa_qty
                    all_pos[i]["stale_momentum_cut_done"] = True
                    all_pos[i]["stale_momentum_cut_at"]   = pos.raw["stale_momentum_cut_at"]
                    all_pos[i]["stale_momentum_cut_qty"]  = cut_qty
                    if new_sl_id:
                        all_pos[i]["sl_order_id"] = new_sl_id
                    break
            _save_all(all_pos)
        except Exception as e:
            logger.error("[monitor] stale_momentum flush JSON error %s: %s", symbol, e)

        # ── Step 5: Notif ─────────────────────────────────────────────────────
        pnl_cut = 0.0
        if mark > 0:
            notional_cut = pos.notional * STALE_MOMENTUM_CUT_PCT
            if pos.side == "BUY":
                pnl_cut = (mark - pos.entry_price) / pos.entry_price * notional_cut
            else:
                pnl_cut = (pos.entry_price - mark) / pos.entry_price * notional_cut
            pnl_cut = round(pnl_cut - notional_cut * TAKER_FEE * 2, 4)

        pnl_str   = f"+{pnl_cut:.2f}" if pnl_cut >= 0 else f"{pnl_cut:.2f}"
        sl_id_str = str(new_sl_id) if new_sl_id else "GAGAL ⚠️"

        self.notify(
            f"✂️ <b>Stale Momentum Cut — {symbol}</b>\n"
            f"  Side   : <b>{'SHORT' if pos.side == 'SELL' else 'LONG'}</b>  "
            f"Age: <b>{age/3600:.1f}h</b>\n"
            f"  Candle : <code>{close_c1:.6f}</code> → <code>{close_c3:.6f}</code>  "
            f"(<b>{move_pct*100:.2f}%</b>)\n"
            f"  ATR-14 : <code>{atr:.6f}</code>  "
            f"Threshold: <code>{threshold_abs:.6f}</code>\n"
            f"  Cut    : <code>{cut_qty}</code> qty (40% original)\n"
            f"  Sisa   : <code>{sisa_qty}</code> qty (60%)\n"
            f"  SL baru: <code>{sl_price:.6f}</code>  algoId <code>{sl_id_str}</code>\n"
            f"  PnL cut: <b>{pnl_str} USDT</b>",
            TOPIC_ORDERS,
        )

    async def _check_volume_reversal(self, pos: MonitoredPosition) -> None:
        mark = self._last_price.get(pos.symbol) or get_mark_price(pos.symbol)
        if not mark:
            return
        if pos.side == "BUY":
            triggered, reason = self.vol_analyzer.check_sell_spike(pos.symbol)
        else:
            triggered, reason = self.vol_analyzer.check_buy_spike(pos.symbol)
        if triggered:
            await self._close_position(pos, mark, reason=f"Volume reversal 📊 {reason}")

    async def _check_btc_volume_reversal(self, pos: MonitoredPosition) -> None:
        mark = self._last_price.get(pos.symbol) or get_mark_price(pos.symbol)
        if not mark:
            return
        if pos.side == "BUY":
            triggered, reason = self.vol_analyzer.check_sell_spike("BTCUSDT")
        else:
            triggered, reason = self.vol_analyzer.check_buy_spike("BTCUSDT")
        if triggered:
            await self._close_position(pos, mark, reason=f"BTC Volume reversal 📊 {reason}")

    # ------------------------------------------------------------------
    # Urgent Circuit Breaker
    # ------------------------------------------------------------------

    async def _check_urgent_cb(self) -> None:
        btc = self._btc_last_price
        if btc <= 0 or self._btc_prev_close <= 0:
            return

        today_utc = time.strftime("%Y-%m-%d", time.gmtime())
        if self._btc_close_date != today_utc:
            self._btc_prev_close = await asyncio.get_event_loop().run_in_executor(
                None, fetch_btc_prev_close
            )
            self._btc_close_date = today_utc

        chg = (btc - self._btc_prev_close) / self._btc_prev_close
        if abs(chg) < URGENT_CB_PCT or is_urgent_cb_triggered()[0]:
            return

        direction   = "UP"  if chg > 0 else "DOWN"
        banned_side = "BUY" if chg < 0 else "SELL"
        register_urgent_cb(direction, banned_side)

        self.notify(
            f"🚨 <b>URGENT Circuit Breaker — BTC {direction}</b>\n"
            f"  BTC prev close : <code>{self._btc_prev_close:.2f}</code>\n"
            f"  BTC now        : <code>{btc:.2f}</code>\n"
            f"  Change         : <b>{chg*100:+.2f}%</b>\n"
            f"  Side banned    : <b>{banned_side}</b> (1 sesi)",
            TOPIC_CB,
        )
        logger.warning("[monitor] Urgent CB BTC %s %.2f%%", direction, chg * 100)

        # Snapshot dulu sebelum loop — posisi bisa hilang saat _close_position
        snapshot = list(self.positions.items())
        for sym, pos in snapshot:
            # Guard: posisi mungkin sudah di-cleanup oleh iterasi sebelumnya
            if sym not in self.positions:
                continue
            mark = self._last_price.get(sym) or get_mark_price(sym)
            if not mark:
                continue
            if pos.side == banned_side:
                await self._close_position(pos, mark, reason=f"Urgent CB BTC {direction} 🚨")
            else:
                # Posisi berlawanan → extend TP 2x untuk ride momentum
                pos.tp = (pos.entry_price + 2 * abs(pos.tp - pos.entry_price)
                          if pos.side == "BUY" else
                          pos.entry_price - 2 * abs(pos.entry_price - pos.tp))

    # ------------------------------------------------------------------
    # Status text (/monitor command)
    # ------------------------------------------------------------------

    def get_status_text(self) -> str:
        pending = load_pending_positions()

        if not self.positions and not pending:
            return "💤 Tidak ada posisi aktif saat ini."

        lines = []

        if self.positions:
            lines.append(f"👁 <b>Live Monitor — {len(self.positions)} posisi</b>\n")
            for sym, pos in self.positions.items():
                mark    = self._last_price.get(sym) or get_mark_price(sym)
                pnl     = pos.calc_pnl(mark) if mark else 0.0
                pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
                be_tag  = "⚖️ BE" if pos.breakeven_hit else "·"
                wti_tag = f"WTI={pos.wti['wti_pct']:.0f}%" if pos.wti else "WTI=?"
                emoji   = "🟢" if pos.side == "BUY" else "🔴"
                lines.append(
                    f"{emoji} <b>{sym}</b> {be_tag} {wti_tag}\n"
                    f"  Entry: <code>{pos.entry_price}</code>  "
                    f"Mark: <code>{mark or '?'}</code>\n"
                    f"  SL: <code>{pos.sl:.6f}</code>  "
                    f"TP: <code>{pos.tp}</code>\n"
                    f"  PnL: <b>{pnl_str} USDT</b>\n"
                )

        if pending:
            lines.append(f"⏳ <b>{len(pending)} pending order</b> — menunggu fill Binance\n")
            for p in pending:
                sym   = p.get("symbol", "?")
                side  = p.get("side", "?")
                entry = p.get("entry_price", "?")
                ws_px = self._last_price.get(sym)
                px_tag = f"  WS: <code>{ws_px:.4f}</code>" if ws_px else ""
                emoji = "🟢" if side == "BUY" else "🔴"
                lines.append(f"  {emoji} <b>{sym}</b> {side} @ <code>{entry}</code>{px_tag}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Startup notif
    # ------------------------------------------------------------------

    async def _notify_startup(self) -> None:
        pending = load_pending_positions()

        if not self.positions and not pending:
            self.notify(
                "👁 <b>Monitor Live Online</b>\n"
                "Tidak ada posisi aktif.\n"
                "Monitor otomatis aktif saat ada order baru.",
                TOPIC_GENERAL,
            )
            return

        lines = ["👁 <b>Monitor Live Online</b>\n"]

        if self.positions:
            lines.append(f"Memantau <b>{len(self.positions)}</b> posisi:\n")
            for sym, pos in self.positions.items():
                mark    = self._last_price.get(sym) or get_mark_price(sym)
                pnl     = pos.calc_pnl(mark) if mark else 0.0
                pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
                emoji   = "🟢" if pos.side == "BUY" else "🔴"
                wti_str = (
                    f"WTI <b>{pos.wti['wti_pct']:.1f}%</b> "
                    f"{'✅ BTC aktif' if pos.wti['btc_active'] else '⚪ BTC off'}"
                ) if pos.wti else "WTI <i>gagal dihitung</i>"
                lines.append(
                    f"{emoji} <b>{sym}</b>\n"
                    f"  Entry : <code>{pos.entry_price}</code>  Mark: <code>{mark or '?'}</code>\n"
                    f"  SL    : <code>{pos.sl:.6f}</code>  TP: <code>{pos.tp}</code>\n"
                    f"  PnL   : <b>{pnl_str} USDT</b>\n"
                    f"  {wti_str}\n"
                )

        if pending:
            lines.append(f"\n⏳ <b>{len(pending)} pending order</b>:\n")
            for p in pending:
                sym   = p.get("symbol", "?")
                side  = p.get("side", "?")
                entry = p.get("entry_price", "?")
                emoji = "🟢" if side == "BUY" else "🔴"
                lines.append(f"  {emoji} <b>{sym}</b> {side} @ <code>{entry}</code>")

        lines.append("\n✅ Trailing stop, breakeven & reversal aktif")
        self.notify("\n".join(lines), TOPIC_GENERAL)
