import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import requests
import websockets
from websockets.exceptions import ConnectionClosed

from volume_analyzer import VolumeAnalyzer
from wti_crypto      import get_wti
from risk_manager    import register_sl
from order.paper_executor import (
    activate_pending_position,
    cancel_pending_position,
    execute_partial_tp,
    check_partial_tp,
    PARTIAL_TP_RR,
    PARTIAL_TP_PCT,
)

logger = logging.getLogger(__name__)

BINANCE_WS_BASE     = "wss://fstream.binance.com/stream"
BINANCE_FUTURES_URL = "https://fapi.binance.com"

RECONNECT_BASE = 2
RECONNECT_MAX  = 30
BREAKEVEN_RR   = 1

# BTC price reversal config
BTC_REVERSAL_PCT    = 0.015
BTC_REVERSAL_WINDOW = 60

# Pending order timeout: cancel jika tidak ter-fill dalam X menit
PENDING_TIMEOUT_MIN = 60


def _find_positions_file() -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "paper_positions.json"),
        "/home/ec2-user/crypto/paper_positions.json",
        os.path.join(os.path.dirname(__file__), "positions.json"),
    ]
    for p in candidates:
        if os.path.exists(os.path.abspath(p)):
            return os.path.abspath(p)
    return os.path.abspath(candidates[0])

POSITIONS_FILE = _find_positions_file()


def load_open_positions() -> List[dict]:
    """Load posisi dengan status 'open' (sudah filled)."""
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE) as f:
                all_pos = json.load(f)
            return [p for p in all_pos if p.get("status") == "open"]
    except Exception as e:
        logger.error("[monitor] Load positions error: %s", e)
    return []


def load_pending_positions() -> List[dict]:
    """Load posisi dengan status 'pending' (limit belum ter-fill)."""
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE) as f:
                all_pos = json.load(f)
            return [p for p in all_pos if p.get("status") == "pending"]
    except Exception as e:
        logger.error("[monitor] Load pending error: %s", e)
    return []


def save_positions(positions: List[dict]) -> None:
    try:
        all_pos = []
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE) as f:
                all_pos = json.load(f)
        updated_ids = {p.get("paper_id") or p.get("symbol") for p in positions}
        result = []
        for p in all_pos:
            pid = p.get("paper_id") or p.get("symbol")
            if pid in updated_ids:
                updated = next(
                    (x for x in positions if (x.get("paper_id") or x.get("symbol")) == pid), p)
                result.append(updated)
            else:
                result.append(p)
        with open(POSITIONS_FILE, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.error("[monitor] Save positions error: %s", e)


def remove_closed_position(paper_id: str) -> None:
    try:
        if not os.path.exists(POSITIONS_FILE):
            return
        with open(POSITIONS_FILE) as f:
            all_pos = json.load(f)
        remaining = [p for p in all_pos if p.get("paper_id") != paper_id]
        with open(POSITIONS_FILE, "w") as f:
            json.dump(remaining, f, indent=2)
        logger.info("[monitor] Posisi %s dihapus", paper_id)
    except Exception as e:
        logger.error("[monitor] remove_closed error: %s", e)


def append_history(record: dict) -> None:
    hist_file = POSITIONS_FILE.replace("paper_positions", "paper_history")
    try:
        history = []
        if os.path.exists(hist_file):
            with open(hist_file) as f:
                history = json.load(f)
        history.append(record)
        with open(hist_file, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error("[monitor] Append history error: %s", e)


_TICK_CACHE: Dict[str, float] = {}

def get_tick_size(symbol: str) -> float:
    sym = symbol.upper()
    if sym in _TICK_CACHE:
        return _TICK_CACHE[sym]
    try:
        resp = requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/exchangeInfo", timeout=10)
        resp.raise_for_status()
        for s in resp.json().get("symbols", []):
            for f in s.get("filters", []):
                if f["filterType"] == "PRICE_FILTER":
                    _TICK_CACHE[s["symbol"]] = float(f["tickSize"])
        return _TICK_CACHE.get(sym, 0.001)
    except Exception as e:
        logger.warning("[monitor] Gagal fetch tick %s: %s", sym, e)
        return 0.001


def get_mark_price(symbol: str) -> Optional[float]:
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex",
            params={"symbol": symbol.upper()}, timeout=5,
        )
        resp.raise_for_status()
        return float(resp.json().get("markPrice", 0))
    except Exception:
        return None


@dataclass
class PendingPosition:
    """Limit order yang menunggu entry price hit."""
    raw:         dict
    paper_id:    str
    symbol:      str
    side:        str
    entry_price: float
    created_at:  float = field(default_factory=time.time)


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
    partial_tp_done: bool = False
    wti:           Optional[dict] = None
    opened_at:     float = field(default_factory=time.time)

    @property
    def sl_initial(self) -> float:
        return float(self.raw.get("stop_loss", self.sl))

    def calc_pnl(self, mark_price: float) -> float:
        if self.side == "BUY":
            return (mark_price - self.entry_price) / self.entry_price * self.notional
        else:
            return (self.entry_price - mark_price) / self.entry_price * self.notional

    def is_breakeven_triggered(self, mark_price: float) -> bool:
        pnl_pct  = self.calc_pnl(mark_price) / self.notional
        risk_pct = self.risk / self.entry_price
        return pnl_pct >= (BREAKEVEN_RR * risk_pct)

    def update_trailing_sl(self, mark_price: float) -> bool:
        if not self.breakeven_hit:
            return False
        tick = self.tick_size
        if self.side == "BUY":
            trail_offset = self.entry_price - self.sl_initial
            new_sl = round(round((mark_price - trail_offset) / tick) * tick, 8)
            if new_sl > self.sl + tick * 0.9:
                self.sl = new_sl
                return True
        else:
            trail_offset = self.sl_initial - self.entry_price
            new_sl = round(round((mark_price + trail_offset) / tick) * tick, 8)
            if new_sl < self.sl - tick * 0.9:
                self.sl = new_sl
                return True
        return False

    def is_sl_hit(self, price: float) -> bool:
        return price <= self.sl if self.side == "BUY" else price >= self.sl

    def is_tp_hit(self, price: float) -> bool:
        return price >= self.tp if self.side == "BUY" else price <= self.tp

    def is_partial_tp_hit(self, price: float) -> bool:
        if self.partial_tp_done:
            return False
        return check_partial_tp(self.raw, price)


class PositionMonitor:

    def __init__(
        self,
        notify: Optional[Callable[[str], None]] = None,
        paper_mode: bool = True,
        poll_interval: float = 5.0,
    ):
        self.notify        = notify or (lambda msg: None)
        self.paper_mode    = paper_mode
        self.poll_interval = poll_interval

        self.positions:  Dict[str, MonitoredPosition] = {}
        self.pendings:   Dict[str, PendingPosition]   = {}  # paper_id → PendingPosition
        self.vol_analyzer = VolumeAnalyzer(spike_multiplier=15.0)

        self._ws_tasks:  Dict[str, asyncio.Task] = {}
        self._running    = False
        self._last_bucket_check: Dict[str, float] = {}
        self._btc_price_buffer: List[tuple] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        logger.info("[monitor] Starting (paper=%s)", self.paper_mode)
        await self._sync_positions()
        asyncio.create_task(self._sync_loop(),    name="monitor-sync")
        asyncio.create_task(self._pending_loop(), name="monitor-pending")
        self._ensure_btc_stream()
        await self._notify_startup()

    async def stop(self) -> None:
        self._running = False
        for task in self._ws_tasks.values():
            task.cancel()
        await asyncio.gather(*self._ws_tasks.values(), return_exceptions=True)
        logger.info("[monitor] Stopped.")

    # ── Sync loop ─────────────────────────────────────────────────────────

    async def _sync_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.poll_interval)
            await self._sync_positions()

    async def _sync_positions(self) -> None:
        open_positions  = load_open_positions()
        open_symbols    = {p.get("symbol", "").upper() for p in open_positions}
        current_symbols = set(self.positions.keys())

        for pos_data in open_positions:
            sym = pos_data.get("symbol", "").upper()
            if sym and sym not in self.positions:
                await self._init_position(sym, pos_data)

        for sym in current_symbols - open_symbols:
            self._cleanup_position(sym)

        # Sync partial_tp_done dari file (kalau restart)
        for pos_data in open_positions:
            sym = pos_data.get("symbol", "").upper()
            if sym in self.positions:
                self.positions[sym].partial_tp_done = bool(pos_data.get("partial_tp_done", False))
                self.positions[sym].raw = pos_data

    # ── Pending order loop ────────────────────────────────────────────────

    async def _pending_loop(self) -> None:
        """
        Tiap poll_interval, cek:
        1. Posisi pending baru dari file → tambah ke self.pendings dan start WS
        2. Pending yang sudah timeout → cancel
        """
        while self._running:
            await asyncio.sleep(self.poll_interval)
            pending_pos = load_pending_positions()
            pending_ids = {p.get("paper_id") for p in pending_pos}

            # Tambah pending baru
            for pos_data in pending_pos:
                pid = pos_data.get("paper_id")
                if pid and pid not in self.pendings:
                    sym = pos_data.get("symbol", "").upper()
                    self.pendings[pid] = PendingPosition(
                        raw         = pos_data,
                        paper_id    = pid,
                        symbol      = sym,
                        side        = pos_data.get("side", "BUY"),
                        entry_price = float(pos_data.get("entry_price", 0)),
                    )
                    self._start_stream(sym)
                    logger.info("[monitor] Pending order tracked: %s %s @ %.6f",
                                pid, sym, float(pos_data.get("entry_price", 0)))

            # Hapus yang sudah tidak pending
            expired_pids = [pid for pid in self.pendings if pid not in pending_ids]
            for pid in expired_pids:
                del self.pendings[pid]

            # Timeout check
            now = time.time()
            for pid, pend in list(self.pendings.items()):
                age_min = (now - pend.created_at) / 60
                if age_min >= PENDING_TIMEOUT_MIN:
                    logger.info("[monitor] Pending %s timeout setelah %.0f menit — cancel", pid, age_min)
                    cancel_pending_position(pid, reason=f"timeout {age_min:.0f}m", notify_fn=self.notify)
                    del self.pendings[pid]

    # ── Init position ──────────────────────────────────────────────────────

    async def _init_position(self, symbol: str, data: dict) -> None:
        tick  = get_tick_size(symbol)
        entry = float(data.get("entry_price", 0))
        sl    = float(data.get("stop_loss",   0))
        tp    = float(data.get("take_profit", 0))
        side  = data.get("side", "BUY").upper()
        risk  = abs(entry - sl)

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
            partial_tp_done = bool(data.get("partial_tp_done", False)),
            opened_at     = float(data.get("opened_at", time.time())),
        )

        loop = asyncio.get_event_loop()
        wti  = await loop.run_in_executor(None, get_wti, symbol)
        pos.wti = wti

        self.positions[symbol] = pos
        self.vol_analyzer.init_symbol(symbol)
        self._last_bucket_check[symbol] = time.time()
        self._start_stream(symbol)
        self._ensure_btc_stream()

        wti_tag = ""
        if wti:
            wti_tag = (
                f"\n  WTI vs BTC : <b>{wti['wti_pct']:.1f}%</b> "
                f"{'✅ aktif' if wti['btc_active'] else '⚪ off'}"
            )

        self.notify(
            f"👁 <b>Monitor aktif — {symbol}</b>\n"
            f"  Side  : <b>{side}</b>\n"
            f"  Entry : <code>{entry}</code>\n"
            f"  SL    : <code>{sl}</code>\n"
            f"  TP    : <code>{tp}</code>  (RR {data.get('rr', '?')})\n"
            f"  TP30% : <code>{data.get('partial_tp_price', '?')}</code>  @ RR {PARTIAL_TP_RR}"
            f"{wti_tag}"
        )
        logger.info("[monitor] Init %s side=%s entry=%.6f sl=%.6f", symbol, side, entry, sl)

    def _cleanup_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)
        self.vol_analyzer.remove_symbol(symbol)
        self._last_bucket_check.pop(symbol, None)
        # Jangan cancel WS kalau masih ada pending di simbol ini
        still_pending = any(p.symbol == symbol for p in self.pendings.values())
        if not still_pending:
            task = self._ws_tasks.pop(symbol, None)
            if task:
                task.cancel()
        if not self.positions and not self.pendings:
            btc_task = self._ws_tasks.pop("BTCUSDT", None)
            if btc_task:
                btc_task.cancel()
        logger.info("[monitor] Cleanup %s", symbol)

    # ── WebSocket ─────────────────────────────────────────────────────────

    def _start_stream(self, symbol: str) -> None:
        sym = symbol.upper()
        existing = self._ws_tasks.get(sym)
        if existing and not existing.done():
            return
        task = asyncio.create_task(
            self._stream_agg_trade(sym), name=f"monitor-ws-{sym}")
        self._ws_tasks[sym] = task

    def _ensure_btc_stream(self) -> None:
        if not self.positions and not self.pendings:
            return
        existing = self._ws_tasks.get("BTCUSDT")
        if existing and not existing.done():
            return
        self.vol_analyzer.init_symbol("BTCUSDT")
        task = asyncio.create_task(
            self._stream_agg_trade("BTCUSDT"), name="monitor-ws-BTCUSDT")
        self._ws_tasks["BTCUSDT"] = task

    async def _stream_agg_trade(self, symbol: str) -> None:
        url   = f"{BINANCE_WS_BASE}?streams={symbol.lower()}@aggTrade"
        delay = RECONNECT_BASE

        while self._running:
            has_open    = symbol in self.positions
            has_pending = any(p.symbol == symbol for p in self.pendings.values())
            if symbol != "BTCUSDT" and not has_open and not has_pending:
                logger.info("[monitor] WS %s stop — tidak ada posisi/pending", symbol)
                return
            if symbol == "BTCUSDT" and not self.positions and not self.pendings:
                logger.info("[monitor] WS BTC stop — tidak ada posisi")
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
                            import json as _json
                            msg  = _json.loads(raw)
                            data = msg.get("data", msg)
                            await self._on_agg_trade(symbol, data)
                        except Exception as e:
                            logger.warning("[monitor] Dispatch error %s: %s", symbol, e)

            except asyncio.CancelledError:
                return
            except RuntimeError as e:
                # Event loop closed saat shutdown — exit bersih tanpa traceback
                if "event loop" in str(e).lower():
                    return
                logger.warning("[monitor] WS RuntimeError %s: %s", symbol, e)
            except ConnectionClosed as e:
                logger.warning("[monitor] WS closed %s: %s", symbol, e)
            except Exception as e:
                logger.warning("[monitor] WS error %s: %s", symbol, e)

            if not self._running:
                return
            try:
                await asyncio.sleep(delay)
            except RuntimeError:
                # Event loop sudah closed — exit bersih
                return
            delay = min(delay * 2, RECONNECT_MAX)

    # ── Trade handler ─────────────────────────────────────────────────────

    async def _on_agg_trade(self, symbol: str, data: dict) -> None:
        price          = float(data.get("p", 0))
        qty            = float(data.get("q", 0))
        is_buyer_maker = bool(data.get("m", False))

        if price <= 0 or qty <= 0:
            return

        self.vol_analyzer.feed(symbol, price, qty, is_buyer_maker)

        if symbol == "BTCUSDT":
            now = time.time()
            self._btc_price_buffer.append((now, price))
            cutoff = now - BTC_REVERSAL_WINDOW
            self._btc_price_buffer = [(t, p) for t, p in self._btc_price_buffer if t >= cutoff]
            for sym, pos in list(self.positions.items()):
                if pos.wti and pos.wti.get("btc_active"):
                    await self._evaluate_btc_reversal(pos, price)
        else:
            # Cek pending orders untuk simbol ini
            for pid, pend in list(self.pendings.items()):
                if pend.symbol == symbol:
                    await self._check_pending_fill(pend, price)

            # Evaluasi posisi open
            pos = self.positions.get(symbol)
            if pos:
                await self._evaluate_position(pos, price)

        # Volume reversal tiap 60 detik
        now = time.time()
        if symbol != "BTCUSDT":
            last = self._last_bucket_check.get(symbol, 0)
            if now - last >= 60:
                self._last_bucket_check[symbol] = now
                pos = self.positions.get(symbol)
                if pos:
                    await self._check_volume_reversal(pos)
        else:
            last = self._last_bucket_check.get("BTCUSDT", 0)
            if now - last >= 60:
                self._last_bucket_check["BTCUSDT"] = now
                for sym, pos in list(self.positions.items()):
                    if pos.wti and pos.wti.get("btc_active"):
                        await self._check_btc_volume_reversal(pos)

    # ── Pending fill check ─────────────────────────────────────────────────

    async def _check_pending_fill(self, pend: PendingPosition, price: float) -> None:
        """
        Cek apakah harga WS sudah menyentuh entry price limit order.
        BUY: harga turun ke entry atau di bawah → fill
        SELL: harga naik ke entry atau di atas → fill
        """
        if pend.side == "BUY":
            hit = price <= pend.entry_price
        else:
            hit = price >= pend.entry_price

        if hit:
            logger.info("[monitor] Pending %s HIT @ %.6f (entry=%.6f)",
                        pend.paper_id, price, pend.entry_price)
            ok = activate_pending_position(pend.paper_id, price, notify_fn=self.notify)
            if ok:
                del self.pendings[pend.paper_id]
                # _sync_positions akan ambil posisi baru di poll berikutnya

    # ── Position evaluation ────────────────────────────────────────────────

    async def _evaluate_position(self, pos: MonitoredPosition, price: float) -> None:
        symbol = pos.symbol

        # Partial TP check (30% di RR 1.5)
        if not pos.partial_tp_done and pos.is_partial_tp_hit(price):
            updated = execute_partial_tp(pos.raw, price, notify_fn=self.notify)
            pos.partial_tp_done = True
            pos.raw = updated
            # Setelah partial TP, geser SL ke entry (breakeven)
            if not pos.breakeven_hit:
                pos.breakeven_hit = True
                pos.sl = pos.entry_price
                self._flush_sl_to_file()
            return

        # Full TP
        if pos.is_tp_hit(price):
            await self._close_position(pos, price, reason="TP tercapai ✅")
            return

        # SL
        if pos.is_sl_hit(price):
            reason = "Trailing SL 🔄" if pos.breakeven_hit else "SL tercapai 🛑"
            await self._close_position(pos, price, reason=reason)
            return

        # Breakeven (jika belum, dan partial TP belum done)
        if not pos.breakeven_hit and pos.is_breakeven_triggered(price):
            pos.breakeven_hit = True
            pos.sl = pos.entry_price
            self._flush_sl_to_file()
            self.notify(
                f"⚖️ <b>Breakeven — {symbol}</b>\n"
                f"  SL → entry: <code>{pos.entry_price}</code> ✅"
            )
            logger.info("[monitor] %s breakeven @ %.6f", symbol, price)

        # Trailing SL
        if pos.breakeven_hit:
            if pos.update_trailing_sl(price):
                self._flush_sl_to_file()
                logger.debug("[monitor] %s trailing SL → %.6f", symbol, pos.sl)

    async def _evaluate_btc_reversal(self, pos: MonitoredPosition, btc_price: float) -> None:
        if not pos.wti or not pos.wti.get("btc_active"):
            return
        if len(self._btc_price_buffer) < 2:
            return
        prices = [p for _, p in self._btc_price_buffer]
        if pos.side == "BUY":
            recent_high = max(prices)
            drop_pct    = (recent_high - btc_price) / recent_high
            if drop_pct >= BTC_REVERSAL_PCT:
                mark = get_mark_price(pos.symbol)
                if mark:
                    await self._close_position(
                        pos, mark,
                        reason=f"BTC price reversal 📉 drop {drop_pct*100:.2f}% dalam {BTC_REVERSAL_WINDOW}s "
                               f"(WTI={pos.wti['wti_pct']:.1f}%)",
                    )
        else:
            recent_low = min(prices)
            rise_pct   = (btc_price - recent_low) / recent_low
            if rise_pct >= BTC_REVERSAL_PCT:
                mark = get_mark_price(pos.symbol)
                if mark:
                    await self._close_position(
                        pos, mark,
                        reason=f"BTC price reversal 📈 naik {rise_pct*100:.2f}% dalam {BTC_REVERSAL_WINDOW}s "
                               f"(WTI={pos.wti['wti_pct']:.1f}%)",
                    )

    async def _check_volume_reversal(self, pos: MonitoredPosition) -> None:
        mark = get_mark_price(pos.symbol)
        if not mark:
            return
        if pos.side == "BUY":
            triggered, reason = self.vol_analyzer.check_sell_spike(pos.symbol)
        else:
            triggered, reason = self.vol_analyzer.check_buy_spike(pos.symbol)
        if triggered:
            await self._close_position(pos, mark, reason=f"Volume reversal 📊 {reason}")

    async def _check_btc_volume_reversal(self, pos: MonitoredPosition) -> None:
        if not pos.wti or not pos.wti.get("btc_active"):
            return
        mark = get_mark_price(pos.symbol)
        if not mark:
            return
        if pos.side == "BUY":
            triggered, reason = self.vol_analyzer.check_sell_spike("BTCUSDT")
        else:
            triggered, reason = self.vol_analyzer.check_buy_spike("BTCUSDT")
        if triggered:
            await self._close_position(
                pos, mark,
                reason=f"BTC volume reversal 📊 (WTI={pos.wti['wti_pct']:.1f}%) {reason}",
            )

    # ── Close position ─────────────────────────────────────────────────────

    async def _close_position(self, pos: MonitoredPosition, close_price: float, reason: str) -> None:
        symbol = pos.symbol
        if symbol not in self.positions:
            return

        TAKER_FEE = 0.0004
        pnl = pos.calc_pnl(close_price)
        pnl -= pos.notional * TAKER_FEE * 2
        pnl  = round(pnl, 4)

        hold_min     = (time.time() - pos.opened_at) / 60
        close_status = "TP" if "TP tercapai" in reason else "SL"

        # Ban logic:
        # ✅ Ban → SL tercapai 🛑 (harga kena stop loss murni)
        # ✅ Ban → Volume reversal 📊 (kondisi market tidak kondusif)
        # ❌ Tidak ban → Trailing SL 🔄 (posisi sempat profit, exit wajar)
        # ❌ Tidak ban → BTC reversal (bukan salah token-nya)
        # ❌ Tidak ban → TP tercapai ✅
        should_ban = (
            "SL tercapai" in reason or
            "Volume reversal" in reason
        )
        if should_ban:
            try:
                register_sl(symbol)
                logger.info("[monitor] Ban registered — %s 4 sesi | reason: %s", symbol, reason)
            except Exception as e:
                logger.warning("[monitor] register_sl error: %s", e)

        raw = pos.raw.copy()
        raw.update({
            "status":        close_status,
            "close_reason":  reason,
            "close_price":   close_price,
            "pnl":           pnl,
            "closed_at":     time.time(),
            "hold_minutes":  round(hold_min, 1),
            "sl_final":      pos.sl,
            "partial_tp_done": pos.partial_tp_done,
        })

        del self.positions[symbol]
        paper_id = pos.raw.get("paper_id", symbol)
        remove_closed_position(paper_id)
        append_history(raw)
        self._cleanup_position(symbol)

        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        emoji   = "✅" if close_status == "TP" else "🛑"
        ban_note = "\n  🚫 Simbol di-ban 4 sesi" if close_status == "SL" else ""

        self.notify(
            f"{emoji} <b>CLOSED — {symbol}</b>\n"
            f"  Reason : <i>{reason}</i>\n"
            f"  Entry  : <code>{pos.entry_price}</code>\n"
            f"  Close  : <code>{close_price:.6f}</code>\n"
            f"  SL     : <code>{pos.sl:.6f}</code>\n"
            f"  PnL    : <b>{pnl_str} USDT</b>\n"
            f"  Hold   : {hold_min:.0f} menit"
            f"  Partial TP: {'✅' if pos.partial_tp_done else '—'}"
            f"{ban_note}"
        )
        logger.info("[monitor] CLOSED %s @ %.6f status=%s pnl=%.4f", symbol, close_price, close_status, pnl)

    # ── Flush SL ───────────────────────────────────────────────────────────

    def _flush_sl_to_file(self) -> None:
        updated = []
        for pos in self.positions.values():
            raw = pos.raw.copy()
            raw["stop_loss"]       = pos.sl
            raw["breakeven_hit"]   = pos.breakeven_hit
            raw["partial_tp_done"] = pos.partial_tp_done
            updated.append(raw)
        if updated:
            save_positions(updated)

    # ── Status text ────────────────────────────────────────────────────────

    def get_status_text(self) -> str:
        lines = []

        if self.pendings:
            lines.append(f"⏳ <b>Pending ({len(self.pendings)})</b>")
            for pid, pend in self.pendings.items():
                age_min = (time.time() - pend.created_at) / 60
                lines.append(
                    f"  {'🟢' if pend.side=='BUY' else '🔴'} <b>{pend.symbol}</b> "
                    f"@ <code>{pend.entry_price}</code>  {age_min:.0f}m  ID:{pid}"
                )

        if not self.positions:
            if not self.pendings:
                return "📭 Tidak ada posisi aktif."
            return "\n".join(lines)

        lines.append(f"\n👁 <b>Aktif ({len(self.positions)})</b>")
        for sym, pos in self.positions.items():
            mark    = get_mark_price(sym)
            pnl     = pos.calc_pnl(mark) if mark else 0.0
            pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
            be_tag  = "⚖️BE" if pos.breakeven_hit else "·"
            ptp_tag = "🎯30%" if pos.partial_tp_done else "·"
            wti_tag = f"WTI={pos.wti['wti_pct']:.0f}%" if pos.wti else "WTI=?"
            emoji   = "🟢" if pos.side == "BUY" else "🔴"
            lines.append(
                f"{emoji} <b>{sym}</b> {be_tag} {ptp_tag} {wti_tag}\n"
                f"  Entry:<code>{pos.entry_price}</code> Mark:<code>{mark or '?'}</code>\n"
                f"  SL:<code>{pos.sl:.6f}</code> TP:<code>{pos.tp}</code>\n"
                f"  PnL: <b>{pnl_str} USDT</b>\n"
            )
        return "\n".join(lines)

    async def _notify_startup(self) -> None:
        mode_tag = "🧪 PAPER" if self.paper_mode else "💰 REAL"
        pending_count = len(load_pending_positions())

        if not self.positions and not pending_count:
            self.notify(
                f"👁 <b>Monitor Online</b> — {mode_tag}\n"
                f"Tidak ada posisi aktif. Monitor siap."
            )
            return

        lines = [f"👁 <b>Monitor Online</b> — {mode_tag}"]
        if pending_count:
            lines.append(f"  ⏳ {pending_count} limit order menunggu fill")
        if self.positions:
            lines.append(f"  📂 {len(self.positions)} posisi aktif dimonitor")
        lines.append(
            f"\nBTC reversal: ≥{BTC_REVERSAL_PCT*100:.1f}% dalam {BTC_REVERSAL_WINDOW}s → close\n"
            f"Partial TP: {PARTIAL_TP_PCT*100:.0f}% @ RR {PARTIAL_TP_RR}:1\n"
            f"Trailing stop & volume reversal aktif."
        )
        self.notify("\n".join(lines))
