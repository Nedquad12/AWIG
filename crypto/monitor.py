# =============================================================
# monitor.py — Real-Time Position Monitor (LIVE ONLY)
#
# Tanggung jawab:
#   1. Baca posisi aktif dari positions.json
#   2. Subscribe WS aggTrade per koin + BTCUSDT
#   3. Per poll (5 detik): tanya Binance API status pending order
#      → FILLED: aktifkan posisi di JSON + notif
#      → CANCELED/EXPIRED/REJECTED: cancel di JSON + notif
#   4. Per tick WS:
#       a. Evaluate posisi open (SL/TP/breakeven/trailing)
#       b. Partial TP @ RR 1.5 via Binance reduce-only market
#   5. Per bucket 60 detik: volume reversal koin + BTC
#   6. Per bucket 15 detik: Urgent Circuit Breaker BTC ≥ 5%
#   7. Semua close via Binance market reduce-only
# =============================================================

import asyncio
import json
import logging
import os
import time
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
BINANCE_FUTURES_URL = "https://fapi.binance.com"

RECONNECT_BASE = 2
RECONNECT_MAX  = 30

BREAKEVEN_RR          = 1.0
PENDING_EXPIRE_SEC    = 2 * 3600
MAX_HOLD_SEC          = 33 * 3600
MAX_FILLS_PER_SESSION = 2
URGENT_CB_PCT         = 0.05
URGENT_CB_BUCKET_SEC  = 15
TAKER_FEE             = 0.0004
PARTIAL_TP_RR         = 1.5
PARTIAL_TP_PCT        = 0.30


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
        self.vol_analyzer = VolumeAnalyzer(spike_multiplier=10.5)

        self._ws_tasks:   Dict[str, asyncio.Task] = {}
        self._running     = False
        self._last_price:  Dict[str, float] = {}
        self._last_bucket: Dict[str, float] = {}
        self._filling:     Set[str] = set()

        self._btc_prev_close:   float = 0.0
        self._btc_last_price:   float = 0.0
        self._urgent_cb_bucket: float = 0.0
        self._btc_close_date:   str   = ""

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
        await self._sync_positions()
        await self._notify_startup()
        asyncio.create_task(self._sync_loop(), name="monitor-sync")

    async def stop(self) -> None:
        self._running = False
        for task in self._ws_tasks.values():
            task.cancel()
        self._ws_tasks.clear()

    # ------------------------------------------------------------------
    # Sync loop
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.poll_interval)
            await self._sync_positions()

    async def _sync_positions(self) -> None:
        cancel_expired_pending(notify_fn=self.notify)
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

        if self.positions:
            self._flush_sl_to_file()

    # ------------------------------------------------------------------
    # Pending fill — poll Binance API
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
                    logger.info("[monitor] Binance FILLED %s %s @ %.6f", order_id, sym, fill_price)

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
                logger.warning("[monitor] Gagal cek Binance order %s %s: %s", sym, binance_oid, e)

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
    # Close posisi — Binance market reduce-only
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

        loop = asyncio.get_event_loop()
        try:
            from order.executor import cancel_all_open_orders, close_position_market
            await loop.run_in_executor(None, cancel_all_open_orders, symbol)
            await loop.run_in_executor(None, close_position_market, symbol, pos.side, pos.qty)
            logger.info("[monitor] Close Binance %s qty=%.6f @ %.6f", symbol, pos.qty, price)
        except Exception as e:
            logger.error("[monitor] Gagal close Binance %s: %s", symbol, e)
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
    # WebSocket streams
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
    # Partial TP @ RR 1.5 — 30% close via Binance
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

        partial_qty      = round(pos.qty * PARTIAL_TP_PCT, 6)
        partial_notional = pos.notional * PARTIAL_TP_PCT
        if pos.side == "BUY":
            partial_pnl = (price - entry) / entry * partial_notional
        else:
            partial_pnl = (entry - price) / entry * partial_notional
        partial_pnl = round(partial_pnl - partial_notional * TAKER_FEE * 2, 4)

        loop = asyncio.get_event_loop()
        try:
            from order.executor import close_position_market
            await loop.run_in_executor(None, close_position_market, pos.symbol, pos.side, partial_qty)
        except Exception as e:
            logger.error("[monitor] Partial TP close gagal %s: %s", pos.symbol, e)
            self.notify(f"🚨 <b>{pos.symbol}</b> — Partial TP gagal: <code>{e}</code>", TOPIC_ERROR)
            return

        pos.qty      = round(pos.qty * (1 - PARTIAL_TP_PCT), 6)
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
    # Volume reversal
    # ------------------------------------------------------------------

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

        for sym, pos in list(self.positions.items()):
            mark = self._last_price.get(sym) or get_mark_price(sym)
            if not mark:
                continue
            if pos.side == banned_side:
                await self._close_position(pos, mark, reason=f"Urgent CB BTC {direction} 🚨")
            else:
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
