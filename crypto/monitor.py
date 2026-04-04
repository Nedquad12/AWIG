import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import requests
import websockets
from websockets.exceptions import ConnectionClosed

from volume_analyzer import VolumeAnalyzer
from wti_crypto      import get_wti

logger = logging.getLogger(__name__)

BINANCE_WS_BASE     = "wss://fstream.binance.com/stream"
BINANCE_FUTURES_URL = "https://fapi.binance.com"

RECONNECT_BASE = 2
RECONNECT_MAX  = 30

BREAKEVEN_RR = 1


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
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE) as f:
                all_pos = json.load(f)
            return [p for p in all_pos if p.get("status") == "open"]
    except Exception as e:
        logger.error("[monitor] Load positions error: %s", e)
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
                # Ganti dengan versi terbaru
                updated = next(
                    (x for x in positions if (x.get("paper_id") or x.get("symbol")) == pid),
                    p,
                )
                result.append(updated)
            else:
                result.append(p)

        with open(POSITIONS_FILE, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.error("[monitor] Save positions error: %s", e)


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
                    tick = float(f["tickSize"])
                    _TICK_CACHE[s["symbol"]] = tick
        return _TICK_CACHE.get(sym, 0.001)
    except Exception as e:
        logger.warning("[monitor] Gagal fetch tick size %s: %s", sym, e)
        return 0.001


def get_mark_price(symbol: str) -> Optional[float]:
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex",
            params={"symbol": symbol.upper()},
            timeout=5,
        )
        resp.raise_for_status()
        return float(resp.json().get("markPrice", 0))
    except Exception:
        return None
    
    
@dataclass
class MonitoredPosition:
    """Extend data posisi dengan state monitor."""
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

    @property
    def sl_initial(self) -> float:
        """SL awal (sebelum trailing) — dari data raw."""
        return float(self.raw.get("stop_loss", self.sl))

    def calc_pnl(self, mark_price: float) -> float:
        if self.side == "BUY":
            return (mark_price - self.entry_price) / self.entry_price * self.notional
        else:
            return (self.entry_price - mark_price) / self.entry_price * self.notional

    def is_breakeven_triggered(self, mark_price: float) -> bool:
        """Profit ≥ 0.8× risk."""
        pnl_pct = self.calc_pnl(mark_price) / self.notional
        risk_pct = self.risk / self.entry_price
        return pnl_pct >= (BREAKEVEN_RR * risk_pct)

    def update_trailing_sl(self, mark_price: float) -> bool:
        """
        Geser SL 1 tick setiap harga bergerak 1 tick ke arah profit.
        Hanya aktif setelah breakeven.
        Return True jika SL berubah.
        """
        if not self.breakeven_hit:
            return False

        tick = self.tick_size

        if self.side == "BUY":
            trail_offset = self.entry_price - self.sl_initial
            new_sl = round(mark_price - trail_offset, 8)
            new_sl = round(round(new_sl / tick) * tick, 8)
            if new_sl > self.sl + tick * 0.9:  
                self.sl = new_sl
                return True

        else:  # SELL
            trail_offset = self.sl_initial - self.entry_price
            new_sl = round(mark_price + trail_offset, 8)
            new_sl = round(round(new_sl / tick) * tick, 8)
            if new_sl < self.sl - tick * 0.9:  
                self.sl = new_sl
                return True

        return False

    def is_sl_hit(self, price: float) -> bool:
        if self.side == "BUY":
            return price <= self.sl
        else:
            return price >= self.sl

    def is_tp_hit(self, price: float) -> bool:
        if self.side == "BUY":
            return price >= self.tp
        else:
            return price <= self.tp

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
        self.vol_analyzer = VolumeAnalyzer(spike_multiplier=3.0)

        self._ws_tasks:   Dict[str, asyncio.Task] = {}
        self._running     = False
        self._last_bucket_check: Dict[str, float] = {}   

    async def start(self) -> None:
        self._running = True
        logger.info("[monitor] Starting PositionMonitor (paper=%s)...", self.paper_mode)

        await self._sync_positions()

        asyncio.create_task(self._sync_loop(), name="monitor-sync")

        self._ensure_btc_stream()

        await self._notify_startup()

    async def stop(self) -> None:
        self._running = False
        for task in self._ws_tasks.values():
            task.cancel()
        await asyncio.gather(*self._ws_tasks.values(), return_exceptions=True)
        logger.info("[monitor] Stopped.")

    async def _sync_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.poll_interval)
            await self._sync_positions()

    async def _sync_positions(self) -> None:
        open_positions = load_open_positions()
        open_symbols   = {p.get("symbol", "").upper() for p in open_positions}
        current_symbols = set(self.positions.keys())

        for pos_data in open_positions:
            sym = pos_data.get("symbol", "").upper()
            if sym and sym not in self.positions:
                await self._init_position(sym, pos_data)

        for sym in current_symbols - open_symbols:
            self._cleanup_position(sym)

        if self.positions:
            self._flush_sl_to_file()

    async def _init_position(self, symbol: str, data: dict) -> None:
        """Init monitor untuk posisi baru."""
        tick  = get_tick_size(symbol)
        entry = float(data.get("entry_price", 0))
        sl    = float(data.get("stop_loss",   0))
        tp    = float(data.get("take_profit", 0))
        side  = data.get("side", "BUY").upper()
        risk  = abs(entry - sl)

        pos = MonitoredPosition(
            raw         = data,
            symbol      = symbol,
            side        = side,
            entry_price = entry,
            sl          = sl,
            tp          = tp,
            qty         = float(data.get("qty", 0)),
            notional    = float(data.get("notional", 0)),
            tick_size   = tick,
            risk        = risk,
            opened_at   = float(data.get("opened_at", time.time())),
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
                f"{'✅ BTC reversal aktif' if wti['btc_active'] else '⚪ BTC diabaikan'}"
            )

        self.notify(
            f"👁 <b>Monitor aktif — {symbol}</b>\n"
            f"  Side  : <b>{side}</b>\n"
            f"  Entry : <code>{entry}</code>\n"
            f"  SL    : <code>{sl}</code>\n"
            f"  TP    : <code>{tp}</code>\n"
            f"  Risk  : <code>{risk:.6f}</code>"
            f"{wti_tag}"
        )
        logger.info("[monitor] Init %s side=%s entry=%.6f sl=%.6f WTI=%s",
                    symbol, side, entry, sl, wti["wti_pct"] if wti else "N/A")

    def _cleanup_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)
        self.vol_analyzer.remove_symbol(symbol)
        self._last_bucket_check.pop(symbol, None)
        task = self._ws_tasks.pop(symbol, None)
        if task:
            task.cancel()
        if not self.positions:
            btc_task = self._ws_tasks.pop("BTCUSDT", None)
            if btc_task:
                btc_task.cancel()
        logger.info("[monitor] Cleanup %s", symbol)
        
    def _start_stream(self, symbol: str) -> None:
        sym = symbol.upper()
        existing = self._ws_tasks.get(sym)
        if existing and not existing.done():
            return
        task = asyncio.create_task(
            self._stream_agg_trade(sym),
            name=f"monitor-ws-{sym}",
        )
        self._ws_tasks[sym] = task

    def _ensure_btc_stream(self) -> None:
        if not self.positions:
            return
        existing = self._ws_tasks.get("BTCUSDT")
        if existing and not existing.done():
            return
        # Init volume analyzer untuk BTC
        self.vol_analyzer.init_symbol("BTCUSDT")
        task = asyncio.create_task(
            self._stream_agg_trade("BTCUSDT"),
            name="monitor-ws-BTCUSDT",
        )
        self._ws_tasks["BTCUSDT"] = task

    async def _stream_agg_trade(self, symbol: str) -> None:
        url   = f"{BINANCE_WS_BASE}?streams={symbol.lower()}@aggTrade"
        delay = RECONNECT_BASE

        while self._running:
            if symbol != "BTCUSDT" and symbol not in self.positions:
                logger.info("[monitor] WS %s stop — posisi tidak ada", symbol)
                return
            if symbol == "BTCUSDT" and not self.positions:
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
            except ConnectionClosed as e:
                logger.warning("[monitor] WS closed %s: %s", symbol, e)
            except Exception as e:
                logger.warning("[monitor] WS error %s: %s", symbol, e)

            if not self._running:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX)

    async def _on_agg_trade(self, symbol: str, data: dict) -> None:
        price           = float(data.get("p", 0))
        qty             = float(data.get("q", 0))
        is_buyer_maker  = bool(data.get("m", False))

        if price <= 0 or qty <= 0:
            return

        self.vol_analyzer.feed(symbol, price, qty, is_buyer_maker)

        if symbol != "BTCUSDT":
            pos = self.positions.get(symbol)
            if pos:
                await self._evaluate_position(pos, price)

        elif symbol == "BTCUSDT":
            for sym, pos in list(self.positions.items()):
                if pos.wti and pos.wti.get("btc_active"):
                    await self._evaluate_btc_reversal(pos, price)
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

    async def _evaluate_position(self, pos: MonitoredPosition, price: float) -> None:
        symbol = pos.symbol

        if pos.is_tp_hit(price):
            await self._close_position(pos, price, reason="TP tercapai ✅")
            return

        if pos.is_sl_hit(price):
            reason = "Trailing SL 🔄" if pos.breakeven_hit else "SL tercapai 🛑"
            await self._close_position(pos, price, reason=reason)
            return

        if not pos.breakeven_hit and pos.is_breakeven_triggered(price):
            pos.breakeven_hit = Truey
            pos.sl = pos.entry_price
            self._flush_sl_to_file()
            self.notify(
                f"⚖️ <b>Breakeven — {symbol}</b>\n"
                f"  SL digeser ke entry: <code>{pos.entry_price}</code>\n"
                f"  Trailing stop aktif ✅"
            )
            logger.info("[monitor] %s breakeven hit @ %.6f", symbol, price)

        if pos.breakeven_hit:
            changed = pos.update_trailing_sl(price)
            if changed:
                self._flush_sl_to_file()
                logger.debug("[monitor] %s trailing SL → %.6f", symbol, pos.sl)

    async def _evaluate_btc_reversal(self, pos: MonitoredPosition, btc_price: float) -> None:
        pass
    async def _check_volume_reversal(self, pos: MonitoredPosition) -> None:
        symbol = pos.symbol
        mark   = get_mark_price(symbol)
        if mark is None:
            return

        if pos.side == "BUY":
            triggered, reason = self.vol_analyzer.check_sell_spike(symbol)
        else:
            triggered, reason = self.vol_analyzer.check_buy_spike(symbol)

        if triggered:
            await self._close_position(pos, mark, reason=f"Volume reversal 📊 {reason}")

    async def _check_btc_volume_reversal(self, pos: MonitoredPosition) -> None:
        if not pos.wti or not pos.wti.get("btc_active"):
            return

        mark = get_mark_price(pos.symbol)
        if mark is None:
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

    async def _close_position(
        self,
        pos: MonitoredPosition,
        close_price: float,
        reason: str,
    ) -> None:
        symbol = pos.symbol
        if symbol not in self.positions:
            return  
        TAKER_FEE = 0.0004
        pnl = pos.calc_pnl(close_price)
        pnl -= pos.notional * TAKER_FEE * 2
        pnl  = round(pnl, 4)

        hold_min = (time.time() - pos.opened_at) / 60

        raw = pos.raw.copy()
        raw["status"]       = "closed"
        raw["close_reason"] = reason
        raw["close_price"]  = close_price
        raw["pnl"]          = pnl
        raw["closed_at"]    = time.time()
        raw["hold_minutes"] = round(hold_min, 1)
        raw["sl_final"]     = pos.sl   
        del self.positions[symbol]
        save_positions([raw])
        append_history(raw)

        self._cleanup_position(symbol)

        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        emoji   = "✅" if pnl >= 0 else "🛑"

        self.notify(
            f"{emoji} <b>CLOSED — {symbol}</b>\n"
            f"  Reason : <i>{reason}</i>\n"
            f"  Entry  : <code>{pos.entry_price}</code>\n"
            f"  Close  : <code>{close_price:.6f}</code>\n"
            f"  SL hit : <code>{pos.sl:.6f}</code>\n"
            f"  PnL    : <b>{pnl_str} USDT</b>\n"
            f"  Hold   : {hold_min:.0f} menit"
        )
        logger.info("[monitor] CLOSED %s @ %.6f reason=%s pnl=%.4f",
                    symbol, close_price, reason, pnl)

    def _flush_sl_to_file(self) -> None:
        """Update stop_loss di paper_positions.json dengan nilai trailing terbaru."""
        updated = []
        for pos in self.positions.values():
            raw = pos.raw.copy()
            raw["stop_loss"]     = pos.sl
            raw["breakeven_hit"] = pos.breakeven_hit
            updated.append(raw)
        if updated:
            save_positions(updated)

    def get_status_text(self) -> str:
        if not self.positions:
            return "📭 Tidak ada posisi yang dimonitor."

        lines = [f"👁 <b>Monitor aktif — {len(self.positions)} posisi</b>\n"]
        for sym, pos in self.positions.items():
            mark = get_mark_price(sym)
            pnl  = pos.calc_pnl(mark) if mark else 0.0
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
        return "\n".join(lines)


    async def _notify_startup(self) -> None:
        """
        Kirim notif startup dengan ringkasan semua posisi aktif + WTI.
        Dipanggil sekali saat bot start.
        """
        mode_tag = "🧪 PAPER" if self.paper_mode else "💰 REAL"

        if not self.positions:
            self.notify(
                f"👁 <b>Monitor Bot Online</b> — {mode_tag}\n"
                f"Tidak ada posisi aktif saat ini.\n"
                f"Monitor akan otomatis aktif saat ada posisi baru."
            )
            return

        lines = [f"👁 <b>Monitor Bot Online</b> — {mode_tag}\n"]
        lines.append(f"Memantau <b>{len(self.positions)}</b> posisi aktif:\n")

        for sym, pos in self.positions.items():
            mark = get_mark_price(sym)
            pnl  = pos.calc_pnl(mark) if mark else 0.0
            pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
            emoji = "🟢" if pos.side == "BUY" else "🔴"

            if pos.wti:
                wti_str = (
                    f"WTI <b>{pos.wti['wti_pct']:.1f}%</b> "
                    f"{'✅ BTC aktif' if pos.wti['btc_active'] else '⚪ BTC off'}"
                )
            else:
                wti_str = "WTI <i>gagal dihitung</i>"

            lines.append(
                f"{emoji} <b>{sym}</b>\n"
                f"  Entry : <code>{pos.entry_price}</code>  "
                f"Mark: <code>{mark or '?'}</code>\n"
                f"  SL    : <code>{pos.sl:.6f}</code>  "
                f"TP: <code>{pos.tp}</code>\n"
                f"  PnL   : <b>{pnl_str} USDT</b>\n"
                f"  {wti_str}\n"
            )

        lines.append("Trailing stop & reversal aktif. Gunakan /monstatus untuk update.")
        self.notify("\n".join(lines))
