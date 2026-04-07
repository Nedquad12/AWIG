# =============================================================
# hft/monitor.py — Real-Time Position Monitor
#
# Tanggung jawab:
#   1. Baca posisi aktif dari paper_positions.json
#   2. Subscribe WS aggTrade untuk setiap koin + BTCUSDT
#   3. Per update harga:
#       a. Trailing stop — geser SL 1 tick setelah breakeven
#       b. Breakeven — trigger saat profit ≥ 1.0× risk (RR 1:1)
#   4. Per bucket 60 detik:
#       a. Volume reversal koin — sell/buy spike ≥ 3× ATR
#       b. Volume reversal BTC  — jika WTI ≥ 50%
#   5. Per bucket 15 detik (BTCUSDT):
#       a. Urgent Circuit Breaker — BTC naik/turun ≥ 5% dari close 1D kemarin
#          → Close paksa posisi berlawanan, TP 2× posisi searah
#          → Ban side berlawanan 1 sesi di scheduler & buatorder
#   6. Eksekusi close via paper_positions.json update
#      (paper mode) atau Binance API (real mode)
#   7. Notifikasi Telegram untuk setiap event
# =============================================================

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
from risk_manager    import register_sl, register_urgent_cb, is_urgent_cb_triggered

logger = logging.getLogger(__name__)

# ── Topic IDs (harus sama dengan scheduler.py) ───────────────────────────────
TOPIC_CB      = 6
TOPIC_ORDERS  = 2
TOPIC_ERROR   = 10
TOPIC_GENERAL = 88

BINANCE_WS_BASE     = "wss://fstream.binance.com/stream"
BINANCE_FUTURES_URL = "https://fapi.binance.com"

# Reconnect
RECONNECT_BASE = 2
RECONNECT_MAX  = 30

# Breakeven trigger: profit ≥ 1.0 × risk (RR 1:1)
BREAKEVEN_RR = 1.0

# Pending order expire: 2 jam tidak kena → cancel otomatis
PENDING_EXPIRE_SEC = 2 * 3600

# Max hold time: 33 jam sejak filled → close paksa (mau untung atau rugi)
MAX_HOLD_SEC = 33 * 3600

# Max filled per sesi sebelum pending sisanya di-cancel
MAX_FILLS_PER_SESSION = 2

# Urgent Circuit Breaker
URGENT_CB_PCT        = 0.05   # BTC naik/turun ≥ 5% dari close 1D kemarin
URGENT_CB_BUCKET_SEC = 15     # cek tiap 15 detik


# ------------------------------------------------------------------
# Baca / tulis paper_positions.json
# ------------------------------------------------------------------

def _find_positions_file() -> str:
    """Cari paper_positions.json dari beberapa lokasi umum."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "paper_positions.json"),
        "/home/ec2-user/crypto/paper_positions.json",
        os.path.join(os.path.dirname(__file__), "positions.json"),
    ]
    for p in candidates:
        if os.path.exists(os.path.abspath(p)):
            return os.path.abspath(p)
    # Default ke root crypto/
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


def load_pending_positions() -> List[dict]:
    """Load semua posisi yang masih pending (belum filled)."""
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE) as f:
                all_pos = json.load(f)
            return [p for p in all_pos if p.get("status") == "pending"]
    except Exception as e:
        logger.error("[monitor] Load pending positions error: %s", e)
    return []


def save_positions(positions: List[dict]) -> None:
    try:
        # Baca semua (termasuk closed), update yang open
        all_pos = []
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE) as f:
                all_pos = json.load(f)

        # Update by paper_id atau symbol
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


# ------------------------------------------------------------------
# Urgent CB — fetch close 1D kemarin
# ------------------------------------------------------------------

def cancel_expired_pending(notify_fn=None) -> List[str]:
    """
    Cek semua posisi pending di file.
    Jika sudah > PENDING_EXPIRE_SEC (2 jam) sejak opened_at → cancel otomatis.
    Return list symbol yang di-cancel.
    """
    def _notify(msg: str):
        if notify_fn:
            try:
                notify_fn(msg)
            except Exception:
                pass

    cancelled = []
    try:
        if not os.path.exists(POSITIONS_FILE):
            return cancelled

        with open(POSITIONS_FILE) as f:
            all_pos = json.load(f)

        now     = time.time()
        changed = False

        for i, pos in enumerate(all_pos):
            if pos.get("status") != "pending":
                continue

            opened_at = float(pos.get("opened_at", now))
            age       = now - opened_at

            if age >= PENDING_EXPIRE_SEC:
                all_pos[i]["status"]        = "cancelled"
                all_pos[i]["cancel_reason"] = f"expired — tidak ter-fill dalam {int(age/60)} menit"
                all_pos[i]["cancelled_at"]  = int(now)
                changed = True

                sym = pos.get("symbol", "?")
                cancelled.append(sym)

                # Live mode: cancel order di Binance agar tidak ter-fill zombie
                if pos.get("live"):
                    try:
                        from order.executor import cancel_all_open_orders
                        cancel_all_open_orders(sym)
                        logger.info(
                            "[monitor] Live expired — all Binance orders cancelled: %s", sym
                        )
                    except Exception as _ce:
                        logger.warning(
                            "[monitor] Gagal cancel Binance orders %s: %s", sym, _ce
                        )

                logger.info(
                    "[monitor] Pending EXPIRED — %s paper_id=%s age=%.0f menit",
                    sym, pos.get("paper_id", "?"), age / 60,
                )
                _notify(
                    f"⏱ <b>Order Expired — {sym}</b>\n"
                    f"  Entry  : <code>{pos.get('entry_price')}</code>\n"
                    f"  Side   : <b>{pos.get('side')}</b>\n"
                    f"  Reason : tidak ter-fill dalam 2 jam\n"
                    f"  Status : ❌ Cancelled"
                )

        if changed:
            with open(POSITIONS_FILE, "w") as f:
                json.dump(all_pos, f, indent=2)

    except Exception as e:
        logger.error("[monitor] cancel_expired_pending error: %s", e)

    return cancelled


def fetch_btc_prev_close() -> float:
    """
    Fetch close candle 1D kemarin BTCUSDT dari Binance Futures.
    Return float harga close, atau 0.0 jika gagal.
    """
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/klines",
            params={"symbol": "BTCUSDT", "interval": "1d", "limit": 2},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        # raw[-2] = candle kemarin (closed), raw[-1] = candle hari ini (open)
        if len(raw) >= 2:
            prev_close = float(raw[-2][4])  # index 4 = close
            logger.info("[monitor] BTC prev close (1D kemarin): %.2f", prev_close)
            return prev_close
    except Exception as e:
        logger.warning("[monitor] Gagal fetch BTC prev close: %s", e)
    return 0.0


def calc_new_tp(entry: float, tp_old: float, side: str, tick: float) -> float:
    """
    Hitung TP baru = 2× potensi profit dari TP awal.
    Dibulatkan ke atas ke tick size terdekat.
    """
    if side == "BUY":
        offset  = tp_old - entry          # profit range awal
        tp_new  = entry + offset * 2      # 2× profit
        # Bulatkan ke atas
        import math
        tp_new = math.ceil(tp_new / tick) * tick
    else:
        offset  = entry - tp_old
        tp_new  = entry - offset * 2
        import math
        tp_new = math.floor(tp_new / tick) * tick  # SHORT: bulatkan ke bawah (lebih jauh dari entry)
    return round(tp_new, 8)


# ------------------------------------------------------------------
# Fetch tick size dari Binance
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# MonitoredPosition — state per posisi
# ------------------------------------------------------------------

@dataclass
class MonitoredPosition:
    """Extend data posisi dengan state monitor."""
    raw:           dict           # data asli dari paper_positions.json
    symbol:        str
    side:          str            # BUY / SELL
    entry_price:   float
    sl:            float          # SL saat ini (bergerak saat trailing)
    tp:            float
    qty:           float
    notional:      float
    tick_size:     float
    risk:          float          # abs(entry - sl_awal) dalam harga
    breakeven_hit: bool = False
    wti:           Optional[dict] = None
    opened_at:     float = field(default_factory=time.time)
    filled_at:     float = 0.0   # unix timestamp saat limit order filled; 0 = fallback ke opened_at

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
        """Profit ≥ 1.0× risk (RR 1:1)."""
        pnl_pct = self.calc_pnl(mark_price) / self.notional
        risk_pct = self.risk / self.entry_price
        return pnl_pct >= (BREAKEVEN_RR * risk_pct)

    def update_trailing_sl(self, mark_price: float) -> bool:
        """
        Trailing stop — aktif setelah breakeven (RR 1:1).
        SL dimulai dari entry dan naik/turun 1 tick mengikuti harga.

        Logika:
          - trail_offset = risk (jarak entry ke SL awal)
          - BUY:  new_sl = mark_price - risk  (tidak boleh < entry)
          - SELL: new_sl = mark_price + risk  (tidak boleh > entry)

        Contoh BUY: entry=2100, risk=50
          harga naik ke 2150 → SL = 2100 (= entry)
          harga naik ke 2200 → SL = 2150
          harga naik ke 2250 → SL = 2200
        """
        if not self.breakeven_hit:
            return False

        tick = self.tick_size

        if self.side == "BUY":
            new_sl = round(mark_price - self.risk, 8)
            new_sl = round(round(new_sl / tick) * tick, 8)
            # Tidak boleh di bawah entry
            new_sl = max(new_sl, self.entry_price)
            if new_sl > self.sl + tick * 0.9:   # naik minimal 1 tick
                self.sl = new_sl
                return True

        else:  # SELL
            new_sl = round(mark_price + self.risk, 8)
            new_sl = round(round(new_sl / tick) * tick, 8)
            # Tidak boleh di atas entry
            new_sl = min(new_sl, self.entry_price)
            if new_sl < self.sl - tick * 0.9:   # turun minimal 1 tick
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


# ------------------------------------------------------------------
# Position Monitor
# ------------------------------------------------------------------

class PositionMonitor:

    def __init__(
        self,
        notify: Optional[Callable[[str], None]] = None,
        paper_mode: bool = True,
        poll_interval: float = 5.0,
    ):
        self._notify_fn    = notify or (lambda msg: None)
        self.paper_mode    = paper_mode
        self.poll_interval = poll_interval

        self.positions:  Dict[str, MonitoredPosition] = {}   # symbol → pos
        self.vol_analyzer = VolumeAnalyzer(spike_multiplier=10.5)

        self._ws_tasks:   Dict[str, asyncio.Task] = {}
        self._running     = False
        self._last_bucket_check: Dict[str, float] = {}   # symbol → last check ts

        # Urgent CB state
        self._btc_prev_close:     float = 0.0   # close 1D kemarin
        self._btc_last_price:     float = 0.0   # harga BTC terakhir dari WS
        self._urgent_cb_bucket:   float = 0.0   # last urgent CB check timestamp
        self._btc_close_date:     str   = ""    # tanggal saat prev_close di-fetch (UTC)

    def notify(self, msg: str, topic: int = TOPIC_ORDERS) -> None:
        """Kirim notifikasi ke topic yang sesuai."""
        try:
            self._notify_fn(msg)
        except Exception as e:
            logger.warning("[monitor] notify error: %s", e)

    def notify_cb(self, msg: str) -> None:
        """Shortcut notify ke topic CB."""
        self.notify(msg, TOPIC_CB)

    def notify_error(self, msg: str) -> None:
        """Shortcut notify ke topic Error."""
        self.notify(msg, TOPIC_ERROR)

    def notify_general(self, msg: str) -> None:
        """Shortcut notify ke topic General."""
        self.notify(msg, TOPIC_GENERAL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        logger.info("[monitor] Starting PositionMonitor (paper=%s)...", self.paper_mode)

        # Fetch BTC close 1D kemarin untuk Urgent CB
        loop = asyncio.get_event_loop()
        self._btc_prev_close = await loop.run_in_executor(None, fetch_btc_prev_close)
        from datetime import datetime, timezone
        self._btc_close_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Load posisi aktif
        await self._sync_positions()

        # Loop utama: sync posisi baru setiap poll_interval
        asyncio.create_task(self._sync_loop(), name="monitor-sync")

        # Start BTC stream (selalu aktif selama ada posisi)
        self._ensure_btc_stream()

        # Notif startup: ringkasan posisi aktif + WTI
        await self._notify_startup()

    async def stop(self) -> None:
        self._running = False
        for task in self._ws_tasks.values():
            task.cancel()
        await asyncio.gather(*self._ws_tasks.values(), return_exceptions=True)
        logger.info("[monitor] Stopped.")

    # ------------------------------------------------------------------
    # Sync posisi dari file
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.poll_interval)
            await self._sync_positions()

    async def _sync_positions(self) -> None:
        """
        Baca paper_positions.json, tambah/hapus posisi yang berubah.
        """
        # Cancel pending yang sudah expired (> 2 jam tidak kena)
        cancel_expired_pending(notify_fn=self.notify)

        # Cek pending fill via harga WS
        await self._check_pending_fills()

        open_positions = load_open_positions()
        open_symbols   = {p.get("symbol", "").upper() for p in open_positions}
        current_symbols = set(self.positions.keys())

        # Posisi baru → init
        for pos_data in open_positions:
            sym = pos_data.get("symbol", "").upper()
            if sym and sym not in self.positions:
                await self._init_position(sym, pos_data)

        # Posisi yang sudah closed → cleanup
        for sym in current_symbols - open_symbols:
            self._cleanup_position(sym)

        # Update SL tersimpan ke file (mungkin trailing sudah geser)
        if self.positions:
            self._flush_sl_to_file()

    async def _check_pending_fills(self) -> None:
        """
        Cek semua pending order apakah sudah hit entry price.
        Jika filled >= MAX_FILLS_PER_SESSION dari sesi yang sama → cancel sisa pending.
        Dipanggil tiap poll_interval (5 detik) dari _sync_positions.
        """
        from order.paper_executor import (
            activate_pending_position,
            cancel_session_pending,
            count_session_filled,
        )

        pending = load_pending_positions()
        if not pending:
            return

        for pos in pending:
            sym        = pos.get("symbol", "").upper()
            paper_id   = pos.get("paper_id", "")
            side       = pos.get("side", "BUY")
            entry      = float(pos.get("entry_price", 0))
            session_id = pos.get("session_id", "")

            if entry <= 0 or not paper_id:
                continue

            # Ambil mark price
            mark = get_mark_price(sym)
            if not mark:
                continue

            # Cek apakah harga sudah hit entry
            hit = (side == "BUY" and mark <= entry) or (side == "SELL" and mark >= entry)
            if not hit:
                continue

            # Fill posisi ini
            filled = activate_pending_position(paper_id, mark, notify_fn=self.notify)
            if not filled:
                continue

            logger.info("[monitor] Pending filled %s %s @ %.6f", paper_id, sym, mark)

            # Cek berapa yang sudah filled dari sesi ini
            if session_id:
                total_filled = count_session_filled(session_id)
                if total_filled >= MAX_FILLS_PER_SESSION:
                    # Cancel semua sisa pending dari sesi yang sama
                    cancelled = cancel_session_pending(session_id, notify_fn=self.notify)
                    if cancelled:
                        logger.info(
                            "[monitor] Sesi %s sudah %d filled — cancel %d pending: %s",
                            session_id, total_filled, len(cancelled), cancelled,
                        )

    async def _init_position(self, symbol: str, data: dict) -> None:
        """Init monitor untuk posisi baru."""
        from config import PAPER_TRADING_MODE
        pos_is_live    = data.get("live", False)
        config_is_live = not PAPER_TRADING_MODE
        if pos_is_live != config_is_live:
            logger.warning(
                "[monitor] %s — pos.live=%s tapi config live=%s. "
                "Mode berbeda dari saat order dibuat. Skip init posisi ini. "
                "Restart sistem untuk sinkronisasi.",
                symbol, pos_is_live, config_is_live,
            )
            self.notify(
                f"⚠️ <b>Mode mismatch — {symbol}</b>\n"
                f"  Posisi dibuat saat live={pos_is_live}, "
                f"config sekarang live={config_is_live}.\n"
                f"  <i>Restart sistem untuk sinkronisasi.</i>"
            )
            return  # skip posisi ini, jangan monitor

        tick  = get_tick_size(symbol)
        entry = float(data.get("entry_price", 0))
        sl    = float(data.get("stop_loss",   0))
        tp    = float(data.get("take_profit", 0))
        side  = data.get("side", "BUY").upper()

        # risk harus dari SL awal (sebelum trailing), bukan SL yang sudah geser.
        # sl_initial disimpan saat order dibuat; fallback ke sl saat ini jika posisi lama.
        sl_initial    = float(data.get("sl_initial", sl))
        risk          = abs(entry - sl_initial)

        # Restore breakeven_hit dari file — FIX: sebelumnya selalu False setelah restart
        breakeven_hit = bool(data.get("breakeven_hit", False))

        pos = MonitoredPosition(
            raw           = data,
            symbol        = symbol,
            side          = side,
            entry_price   = entry,
            sl            = sl,            # SL terbaru (sudah trailing jika ada)
            tp            = tp,
            qty           = float(data.get("qty", 0)),
            notional      = float(data.get("notional", 0)),
            tick_size     = tick,
            risk          = risk,          # risk dari SL awal, untuk trailing offset
            breakeven_hit = breakeven_hit, # ← FIX: restore dari file
            opened_at     = float(data.get("opened_at", time.time())),
            filled_at     = float(data.get("filled_at", 0)),
        )

        if breakeven_hit:
            logger.info(
                "[monitor] %s restored: breakeven_hit=True, trailing SL=%.6f (dari file)",
                symbol, sl,
            )

        # Hitung WTI (async di executor agar tidak block)
        loop = asyncio.get_event_loop()
        wti  = await loop.run_in_executor(None, get_wti, symbol)
        pos.wti = wti

        self.positions[symbol] = pos
        self.vol_analyzer.init_symbol(symbol)
        self._last_bucket_check[symbol] = time.time()

        # Start WS stream untuk pair ini
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
            f"{be_tag}"
            f"{wti_tag}",
            TOPIC_ORDERS,
        )
        logger.info("[monitor] Init %s side=%s entry=%.6f sl=%.6f be=%s WTI=%s",
                    symbol, side, entry, sl, breakeven_hit, wti["wti_pct"] if wti else "N/A")

    def _cleanup_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)
        self.vol_analyzer.remove_symbol(symbol)
        self._last_bucket_check.pop(symbol, None)
        # Stop WS stream
        task = self._ws_tasks.pop(symbol, None)
        if task:
            task.cancel()
        # Stop BTC stream jika tidak ada posisi lagi
        if not self.positions:
            btc_task = self._ws_tasks.pop("BTCUSDT", None)
            if btc_task:
                btc_task.cancel()
        logger.info("[monitor] Cleanup %s", symbol)

    # ------------------------------------------------------------------
    # WebSocket streams
    # ------------------------------------------------------------------

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
        """
        Stream aggTrade untuk satu symbol.
        Auto-reconnect dengan backoff.
        """
        url   = f"{BINANCE_WS_BASE}?streams={symbol.lower()}@aggTrade"
        delay = RECONNECT_BASE

        while self._running:
            # Stop jika tidak ada posisi yang butuh symbol ini
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

    # ------------------------------------------------------------------
    # Core: proses setiap aggTrade
    # ------------------------------------------------------------------

    async def _on_agg_trade(self, symbol: str, data: dict) -> None:
        price           = float(data.get("p", 0))
        qty             = float(data.get("q", 0))
        is_buyer_maker  = bool(data.get("m", False))

        if price <= 0 or qty <= 0:
            return

        # Feed ke volume analyzer
        self.vol_analyzer.feed(symbol, price, qty, is_buyer_maker)

        # Simpan last BTC price untuk urgent CB
        if symbol == "BTCUSDT":
            self._btc_last_price = price

        # Proses posisi koin (bukan BTC)
        if symbol != "BTCUSDT":
            pos = self.positions.get(symbol)
            if pos:
                await self._evaluate_position(pos, price)

        # Proses reversal BTC untuk semua posisi yang WTI aktif
        elif symbol == "BTCUSDT":
            for sym, pos in list(self.positions.items()):
                if pos.wti and pos.wti.get("btc_active"):
                    await self._evaluate_btc_reversal(pos, price)

        # Cek bucket 60 detik — volume reversal
        now = time.time()
        if symbol != "BTCUSDT":
            last = self._last_bucket_check.get(symbol, 0)
            if now - last >= 60:
                self._last_bucket_check[symbol] = now
                pos = self.positions.get(symbol)
                if pos:
                    await self._check_volume_reversal(pos)
        else:
            # Cek BTC volume setiap 60 detik
            last = self._last_bucket_check.get("BTCUSDT", 0)
            if now - last >= 60:
                self._last_bucket_check["BTCUSDT"] = now
                for sym, pos in list(self.positions.items()):
                    if pos.wti and pos.wti.get("btc_active"):
                        await self._check_btc_volume_reversal(pos)

            # Cek Urgent CB setiap 15 detik (hanya dari stream BTCUSDT)
            if now - self._urgent_cb_bucket >= URGENT_CB_BUCKET_SEC:
                self._urgent_cb_bucket = now
                await self._check_urgent_cb()

    # ------------------------------------------------------------------
    # Evaluate posisi per tick
    # ------------------------------------------------------------------

    async def _evaluate_position(self, pos: MonitoredPosition, price: float) -> None:
        symbol = pos.symbol

        # ── Max Hold (33 jam sejak filled) ────────────────────────
        ref_ts  = pos.filled_at if pos.filled_at > 0 else pos.opened_at
        elapsed = time.time() - ref_ts
        if elapsed >= MAX_HOLD_SEC:
            hold_h = elapsed / 3600
            await self._close_position(pos, price, reason=f"Max Hold {hold_h:.1f}h ⏰")
            return

        # ── TP hit ────────────────────────────────────────────────
        if pos.is_tp_hit(price):
            await self._close_position(pos, price, reason="TP tercapai ✅")
            return

        # ── SL hit (termasuk trailing SL) ─────────────────────────
        if pos.is_sl_hit(price):
            reason = "Trailing SL 🔄" if pos.breakeven_hit else "SL tercapai 🛑"
            await self._close_position(pos, price, reason=reason)
            return

        # ── Breakeven check ───────────────────────────────────────
        if not pos.breakeven_hit and pos.is_breakeven_triggered(price):
            pos.breakeven_hit = True
            # Geser SL ke entry (lokal)
            pos.sl = pos.entry_price
            self._flush_sl_to_file()

            # Live: amend SL Binance ke entry juga
            if pos.raw.get("live"):
                loop = asyncio.get_event_loop()
                try:
                    from order.paper_executor import amend_breakeven_sl
                    updated_raw = await loop.run_in_executor(
                        None,
                        lambda: amend_breakeven_sl(pos.raw, notify_fn=self.notify),
                    )
                    # Update sl_order_id baru di raw agar cancel_all tahu ID-nya
                    pos.raw.update(updated_raw)
                except Exception as e:
                    logger.error("[monitor] amend_breakeven_sl error %s: %s", symbol, e)

            self.notify(
                f"⚖️ <b>Breakeven — {symbol}</b>\n"
                f"  SL digeser ke entry: <code>{pos.entry_price}</code>\n"
                f"  Trailing stop aktif ✅"
                + (" | SL Binance diupdate ✅" if pos.raw.get("live") else ""),
                TOPIC_ORDERS,
            )
            logger.info("[monitor] %s breakeven hit @ %.6f", symbol, price)

        # ── Trailing stop ─────────────────────────────────────────
        if pos.breakeven_hit:
            changed = pos.update_trailing_sl(price)
            if changed:
                self._flush_sl_to_file()
                logger.debug("[monitor] %s trailing SL → %.6f", symbol, pos.sl)

    async def _evaluate_btc_reversal(self, pos: MonitoredPosition, btc_price: float) -> None:
        """Cek apakah BTC price menyentuh SL/TP pos — tidak relevan langsung."""
        # BTC reversal dicek via volume bucket (di _check_btc_volume_reversal)
        # Di sini tidak ada aksi per-tick untuk BTC
        pass

    # ------------------------------------------------------------------
    # Volume reversal checks (setiap 60 detik)
    # ------------------------------------------------------------------

    async def _check_volume_reversal(self, pos: MonitoredPosition) -> None:
        """Cek volume reversal dari koin itu sendiri."""
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
        """Cek volume reversal dari BTC (hanya jika WTI ≥ 50%)."""
        if not pos.wti or not pos.wti.get("btc_active"):
            return

        mark = get_mark_price(pos.symbol)
        if mark is None:
            return

        # LONG → takut BTC sell spike
        if pos.side == "BUY":
            triggered, reason = self.vol_analyzer.check_sell_spike("BTCUSDT")
        else:
            triggered, reason = self.vol_analyzer.check_buy_spike("BTCUSDT")

        if triggered:
            await self._close_position(
                pos, mark,
                reason=f"BTC volume reversal 📊 (WTI={pos.wti['wti_pct']:.1f}%) {reason}",
            )

    # ------------------------------------------------------------------
    # Urgent Circuit Breaker — BTC ±5% dari close 1D kemarin
    # ------------------------------------------------------------------

    async def _check_urgent_cb(self) -> None:
        """
        Cek apakah BTC sudah naik/turun ≥ 5% dari close 1D kemarin.
        Dipanggil tiap 15 detik dari stream BTCUSDT.

        Aksi jika trigger:
        - Close semua posisi berlawanan arah BTC (profit atau tidak)
        - Naikkan TP 2× posisi searah
        - Set flag di risk_manager (ban side berlawanan 1 sesi)
        - Nonaktifkan pengecekan sampai candle 1D berikutnya (00:00 UTC)
        """
        from datetime import datetime, timezone

        # Refresh prev_close jika hari sudah berganti (candle 1D baru)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today_str != self._btc_close_date:
            loop = asyncio.get_event_loop()
            self._btc_prev_close = await loop.run_in_executor(None, fetch_btc_prev_close)
            self._btc_close_date = today_str
            logger.info("[monitor] BTC prev close di-refresh: %.2f", self._btc_prev_close)

        # Belum ada data
        if self._btc_prev_close <= 0 or self._btc_last_price <= 0:
            return

        # Cek apakah sudah trigger hari ini
        already, triggered_dir = is_urgent_cb_triggered()
        if already:
            return

        # Hitung perubahan dari close kemarin
        chg_pct = (self._btc_last_price - self._btc_prev_close) / self._btc_prev_close

        if abs(chg_pct) < URGENT_CB_PCT:
            return

        # Tentukan arah
        btc_up = chg_pct >= URGENT_CB_PCT   # BTC naik ≥5%
        direction    = "UP"   if btc_up else "DOWN"
        close_side   = "SELL" if btc_up else "BUY"   # posisi yang di-close (berlawanan)
        tp_boost_side= "BUY"  if btc_up else "SELL"  # posisi yang TP-nya dinaikkan (searah)
        banned_side  = "SHORT" if btc_up else "LONG"  # side yang di-ban sesi berikutnya

        logger.warning(
            "[monitor] 🚨 URGENT CB — BTC %+.2f%% dari close kemarin %.2f → sekarang %.2f | "
            "close %s, boost TP %s, ban %s",
            chg_pct * 100, self._btc_prev_close, self._btc_last_price,
            close_side, tp_boost_side, banned_side,
        )

        closed_syms   = []
        tp_boosted    = []

        for sym, pos in list(self.positions.items()):
            if pos.side == close_side:
                # Close paksa posisi berlawanan
                mark = get_mark_price(sym)
                close_px = mark if mark else self._btc_last_price
                await self._close_position(
                    pos, close_px,
                    reason=f"Urgent CB 🚨 BTC {direction} {abs(chg_pct)*100:.1f}% — close {close_side}",
                )
                closed_syms.append(sym)

            elif pos.side == tp_boost_side:
                # Naikkan TP 2× untuk posisi searah
                old_tp  = float(pos.raw.get("take_profit", pos.tp))
                new_tp  = calc_new_tp(pos.entry_price, old_tp, pos.side, pos.tick_size)
                pos.tp  = new_tp
                # Update di raw dan flush ke file
                pos.raw["take_profit"] = new_tp
                tp_boosted.append(f"{sym} TP {old_tp:.4f}→{new_tp:.4f}")

        # Flush TP baru ke file
        if tp_boosted:
            self._flush_sl_to_file()

        # Register ke risk_manager — ban side berlawanan 1 sesi
        register_urgent_cb(direction=direction, banned_side=banned_side)

        # Kirim notif
        chg_emoji = "🚀" if btc_up else "💀"
        close_txt = ", ".join(closed_syms) if closed_syms else "tidak ada"
        tp_txt    = "\n  ".join(tp_boosted)  if tp_boosted  else "tidak ada"

        self.notify(
            f"🚨 <b>URGENT Circuit Breaker — BTC {direction}</b>\n"
            f"  BTC kemarin : <code>{self._btc_prev_close:,.2f}</code>\n"
            f"  BTC sekarang: <code>{self._btc_last_price:,.2f}</code>\n"
            f"  Perubahan   : {chg_emoji} <b>{chg_pct*100:+.2f}%</b>\n\n"
            f"  ❌ Di-close  : <b>{close_txt}</b>\n"
            f"  🎯 TP 2×    : <b>{tp_txt if tp_txt else 'tidak ada'}</b>\n\n"
            f"  🚫 Sesi berikutnya: <b>{banned_side} di-ban 1 sesi</b>",
            TOPIC_CB,
        )

    # ------------------------------------------------------------------
    # Close posisi
    # ------------------------------------------------------------------

    async def _close_position(
        self,
        pos: MonitoredPosition,
        close_price: float,
        reason: str,
    ) -> None:
        symbol = pos.symbol
        if symbol not in self.positions:
            return   # sudah di-close sebelumnya

        # Hitung PnL
        TAKER_FEE = 0.0004
        pnl = pos.calc_pnl(close_price)
        pnl -= pos.notional * TAKER_FEE * 2
        pnl  = round(pnl, 4)

        hold_min = (time.time() - pos.opened_at) / 60

        # Tentukan status:
        # ✅ TP       → reason "TP tercapai"
        # ✅ TP       → reason "Trailing SL" (sempat profit, exit wajar → WIN)
        # ⏰ Max Hold → PnL ≥ 0 = TP (WIN), PnL < 0 = SL (LOSS), tidak di-ban
        # 🛑 SL       → reason "SL tercapai" (SL murni)
        # 🛑 SL       → reason "Volume reversal" (kondisi market buruk)
        # 🛑 SL       → reason "BTC volume reversal"
        if "Max Hold" in reason:
            close_status = "TP" if pnl >= 0 else "SL"
            should_ban   = False   # max hold tidak pernah di-ban, biar slot bebas
        elif "TP tercapai" in reason or "Trailing SL" in reason:
            close_status = "TP"
            should_ban   = False
        else:
            close_status = "SL"
            should_ban   = True

        if should_ban:
            try:
                register_sl(symbol)
                logger.info("[monitor] Ban registered — %s 2 sesi | reason: %s", symbol, reason)
            except Exception as e:
                logger.warning("[monitor] register_sl error: %s", e)

        # Update status di file
        raw = pos.raw.copy()
        raw["status"]       = close_status
        raw["close_reason"] = reason
        raw["close_price"]  = close_price
        raw["pnl"]          = pnl
        raw["closed_at"]    = time.time()
        raw["hold_minutes"] = round(hold_min, 1)
        raw["sl_final"]     = pos.sl   # SL akhir setelah trailing

        # Hapus dari active, simpan ke history
        del self.positions[symbol]
        save_positions([raw])
        append_history(raw)

        self._cleanup_position(symbol)

        pnl_str  = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        if "Max Hold" in reason:
            emoji = "⏰✅" if pnl >= 0 else "⏰🛑"
        else:
            emoji = "✅" if close_status == "TP" else "🛑"
        ban_note = "\n  🚫 Simbol di-ban 2 sesi" if should_ban else ""

        self.notify(
            f"{emoji} <b>CLOSED — {symbol}</b>\n"
            f"  Reason : <i>{reason}</i>\n"
            f"  Entry  : <code>{pos.entry_price}</code>\n"
            f"  Close  : <code>{close_price:.6f}</code>\n"
            f"  SL     : <code>{pos.sl:.6f}</code>\n"
            f"  PnL    : <b>{pnl_str} USDT</b>\n"
            f"  Hold   : {hold_min:.0f} menit"
            f"{ban_note}",
            TOPIC_ORDERS,
        )
        logger.info("[monitor] CLOSED %s @ %.6f status=%s pnl=%.4f",
                    symbol, close_price, close_status, pnl)

    # ------------------------------------------------------------------
    # Flush SL terbaru ke file (setelah trailing bergerak)
    # ------------------------------------------------------------------

    def _flush_sl_to_file(self) -> None:
        """Update stop_loss di paper_positions.json dengan nilai trailing terbaru."""
        updated = []
        for pos in self.positions.values():
            raw = pos.raw.copy()
            raw["stop_loss"]     = pos.sl
            raw["breakeven_hit"] = pos.breakeven_hit
            # Simpan sl_initial sekali — agar risk tidak corrupt saat restart
            # (risk = abs(entry - sl_initial), bukan sl trailing yang sudah geser)
            if "sl_initial" not in raw:
                raw["sl_initial"] = pos.raw.get("stop_loss", pos.sl)
            updated.append(raw)
        if updated:
            save_positions(updated)

    # ------------------------------------------------------------------
    # Status summary
    # ------------------------------------------------------------------

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
                f"Monitor akan otomatis aktif saat ada posisi baru.",
                TOPIC_GENERAL,
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
        self.notify("\n".join(lines), TOPIC_GENERAL)
