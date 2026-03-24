# Crypto Futures Trading Bot

Bot trading crypto otomatis berbasis **Telegram**, **Binance Futures Testnet**, **XGBoost ML**, dan **DeepSeek AI** — dengan Stop Loss dan Take Profit otomatis.

---

## Arsitektur & Flow

```
/scan BTCUSDT
     │
     ▼
┌─────────────────────────────────────────────────┐
│ TAHAP 1 — Fetch Data                            │
│   api_binance.py → GET /fapi/v1/klines          │
│   500 candle 15m dari Binance Futures Testnet   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ TAHAP 2 — Hitung 10 Indikator                   │
│   scorer.py + indicators/                       │
│   VSA  FSA  VFA  WCC  SRST                      │
│   RSI  MACD  MA  IP  Tight                      │
│   → Total Score (dengan weight per fitur)       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ TAHAP 3 — Simpan Score History                  │
│   history_manager.py → history/BTCUSDT.json     │
│   Dipakai sebagai data training ML              │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ TAHAP 4 — ML Weight Adjustment                  │
│   ml_engine.py → XGBoost                        │
│   Train dari score history (label: +3 candle)   │
│   Feature importance → update weight per fitur  │
│   Akurasi sebelum vs sesudah dilaporkan         │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ TAHAP 5 — ML Prediction + SL/TP Suggestion      │
│   ml_engine.py                                  │
│   Prediksi arah 3 candle ke depan:              │
│     NAIK | NETRAL | TURUN + confidence          │
│   SL/TP suggestion dari ATR14:                  │
│     SL = ATR14 × 1.5  (dalam %)                 │
│     TP = SL × 2.0     (Risk/Reward default)     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ TAHAP 6 — AI Decision (DeepSeek)                │
│   ai_decision.py                                │
│   Input: OHLCV 20 candle + semua skor +         │
│          hasil ML + SL/TP saran + balance       │
│   Output JSON:                                  │
│     decision:    BUY | SKIP                     │
│     direction:   LONG | SHORT                   │
│     leverage:    1–30x                          │
│     capital_pct: 5–25% modal                    │
│     confidence:  0–100%                         │
│     sl_pct:      % jarak SL dari entry          │
│     tp_pct:      % jarak TP dari entry          │
│     reason:      alasan (bahasa Indonesia)      │
└────────────────────┬────────────────────────────┘
                     │
               decision = BUY?
               ┌──────┴──────┐
              YES             NO
               │               │
               ▼               ▼
┌──────────────────────┐   ⏭ SKIP — kirim
│ TAHAP 7 — Execute    │   notifikasi alasan
│  trader.py           │
│  1. Set leverage     │
│  2. Market entry     │
│  3. Place SL order   │
│     (STOP_MARKET)    │
│  4. Place TP order   │
│     (TAKE_PROFIT_    │
│      MARKET)         │
│  5. Simpan ke        │
│     positions.json   │
└──────────────────────┘
```

---

## Stop Loss & Take Profit

### Cara Kerja

SL dan TP **ditentukan oleh dua lapisan**:

**Lapisan 1 — ML (ATR-based suggestion)**
```
ATR14  = Average True Range 14 period dari candle history
SL%    = (ATR14 / last_price) × 100 × SL_ATR_MULT   (default: × 1.5)
TP%    = SL% × TP_RR_RATIO                           (default: × 2.0)
```
Contoh: ATR = 200 USDT, price = 50.000 USDT → ATR% = 0.4% → SL = 0.6% → TP = 1.2%

**Lapisan 2 — AI Review**
DeepSeek menerima SL/TP suggestion dari ML lalu memutuskan nilai final.
AI boleh memperlebar SL/TP tapi tidak boleh mempersempit di bawah `SL_MIN_PCT`.

### Aturan Enforced (hard limit di kode)

| Aturan | Nilai Default | File |
|--------|--------------|------|
| SL minimum | 0.5% | `config.py → SL_MIN_PCT` |
| SL maximum | 5.0% | `config.py → SL_MAX_PCT` |
| TP minimum | 0.75% | `config.py → TP_MIN_PCT` |
| TP maximum | 15.0% | `config.py → TP_MAX_PCT` |
| Minimum R/R | 1.5x | di-enforce di `ai_decision.py` |

### Cara Aplikasi ke Order

```
LONG:
  SL price = entry × (1 − sl_pct / 100)   → STOP_MARKET SELL
  TP price = entry × (1 + tp_pct / 100)   → TAKE_PROFIT_MARKET SELL

SHORT:
  SL price = entry × (1 + sl_pct / 100)   → STOP_MARKET BUY
  TP price = entry × (1 − tp_pct / 100)   → TAKE_PROFIT_MARKET BUY
```

Binance akan otomatis close posisi ketika mark price menyentuh SL atau TP.

### Saat Close Manual `/close SYMBOL`

1. Bot **cancel semua open order** SL/TP (via `DELETE /fapi/v1/allOpenOrders`)
2. Bot place **market order** untuk flat posisi
3. State lokal dihapus dari `positions.json`

---

## Scaling Leverage & Modal

| Confidence AI | Max Leverage | Max Capital |
|--------------|-------------|------------|
| ≥ 70%        | 30x         | 25%        |
| ≥ 50%        | 15x         | 12.5%      |
| < 50%        | 5x          | 6.25%      |

Semua batas ini bisa diubah di `config.py`.

---

## Setup

### 1. Install dependencies

```bash
cd crypto_bot
pip install -r requirements.txt
```

### 2. Binance Futures Testnet

1. Daftar di: https://testnet.binancefuture.com
2. Login → klik **API Management** → **Create API**
3. Copy `API Key` dan `Secret Key`
4. Isi di `config.py`:
   ```python
   BINANCE_API_KEY    = "xxxx"
   BINANCE_API_SECRET = "xxxx"
   ```

> **Penting:** Pastikan kamu mengaktifkan **Futures Trading** di permission API key.

### 3. Telegram Bot

1. Buka Telegram → cari **@BotFather** → `/newbot`
2. Ikuti instruksi, copy token
3. Cari **@userinfobot** → `/start` → dapat `chat_id` kamu
4. Isi di `config.py`:
   ```python
   TELEGRAM_BOT_TOKEN = "xxxx:xxxx"
   TELEGRAM_CHAT_ID   = 123456789   # chat_id kamu
   TELEGRAM_TOPIC_ID  = None        # None jika bukan supergroup dengan topic
   ```

### 4. DeepSeek API

1. Daftar di: https://platform.deepseek.com
2. Buka **API Keys** → **Create new key**
3. Isi di `config.py`:
   ```python
   DEEPSEEK_API_KEY = "sk-xxxx"
   ```

### 5. Konfigurasi pair yang mau di-trade

Edit `TRADE_PAIRS` di `config.py`:
```python
TRADE_PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    # tambah pair lain sesuai kebutuhan
]
```

### 6. (Opsional) Sesuaikan parameter risk

```python
# config.py

MAX_CAPITAL_PCT    = 0.25   # max 25% modal per trade
MAX_LEVERAGE       = 30     # max leverage 30x

# SL/TP range
SL_MIN_PCT   = 0.5    # SL minimum 0.5%
SL_MAX_PCT   = 5.0    # SL maksimum 5%
TP_MIN_PCT   = 0.75   # TP minimum 0.75%
TP_MAX_PCT   = 15.0   # TP maksimum 15%

# ML ATR multiplier
SL_ATR_MULT  = 1.5    # SL = ATR × 1.5
TP_RR_RATIO  = 2.0    # TP = SL × 2.0
```

### 7. Jalankan bot

```bash
python bot.py
```

---

## Commands Telegram

| Command | Deskripsi |
|---------|-----------|
| `/scan BTCUSDT` | **Full pipeline** — fetch, skor, ML, AI, entry + SL/TP otomatis |
| `/dryrun BTCUSDT` | Full pipeline **tanpa eksekusi order** (safe untuk test) |
| `/skor BTCUSDT` | Hanya hitung skor indikator (cepat ~5 detik) |
| `/scanall` | Scan semua pair di `TRADE_PAIRS` (live) |
| `/dryrunall` | Dry run semua pair |
| `/pos` | Lihat semua posisi terbuka + SL/TP yang aktif |
| `/close BTCUSDT` | Tutup posisi BTCUSDT (cancel SL/TP dulu → market close) |
| `/balance` | Cek saldo USDT Futures Testnet |
| `/pairs` | Lihat daftar pair yang dikonfigurasi |
| `/help` | Tampilkan daftar command |

---

## Struktur File

```
crypto_bot/
│
├── bot.py               ← Entry point Telegram bot
├── config.py            ← Semua konfigurasi (API key, pairs, risk params)
├── pipeline.py          ← Orkestrasi full flow (7 tahap)
│
├── api_binance.py       ← Binance Futures Testnet API
│                          (fetch OHLCV, place order, SL/TP, cancel, balance)
│
├── scorer.py            ← Hitung total score dari semua indikator
├── history_manager.py   ← Simpan/load score history per symbol (untuk ML)
│
├── ml_engine.py         ← XGBoost:
│                          - run_ml_weights(): adjust weight per fitur
│                          - predict_direction(): prediksi 3 candle ke depan
│                          - suggest_sl_tp(): hitung SL/TP dari ATR14
│
├── ai_decision.py       ← DeepSeek AI:
│                          - terima skor + ML prediction + SL/TP saran
│                          - output: BUY/SKIP, leverage, capital, sl_pct, tp_pct
│
├── trader.py            ← Eksekusi order:
│                          - entry market order
│                          - place STOP_MARKET (SL)
│                          - place TAKE_PROFIT_MARKET (TP)
│                          - close_trade() dengan cancel SL/TP dulu
│
├── weight_manager.py    ← Load/save weight per symbol
├── formatter.py         ← Format semua pesan Telegram (HTML)
├── requirements.txt
│
├── indicators/
│   ├── __init__.py
│   ├── vsa.py           Volume Super Analysis
│   ├── rsi.py           Relative Strength Index (14)
│   ├── macd.py          MACD (12, 26, 9)
│   ├── ma.py            Moving Average (20/60/120/200)
│   ├── ip.py            Indikator Poin (MACD + Stoch × 3 TF)
│   ├── fsa.py           Frekuensi Super Analysis
│   ├── vfa.py           Volume Frequency Analysis
│   ├── wcc.py           Wick Candle Change
│   ├── srst.py          Support & Resistance Smart Tool
│   ├── sr.py            S&R detector (Donchian)
│   └── tight.py         Very Tight / Tight scan
│
├── cache/               (auto) OHLCV sementara
├── history/             (auto) Score history JSON per symbol
├── weights/             (auto) ML weight per symbol
├── logs/                (auto) Log file
└── positions.json       (auto) State posisi terbuka + SL/TP info
```

---

## Indikator & Skor

| Indikator | Range | Logika |
|-----------|-------|--------|
| VSA  | -2 → +2 | Avg volume 7h vs 30h |
| FSA  | -1 → +2 | Avg transaksi 7h vs 30h |
| VFA  | -3 → +3 | % change volume vs transaksi harian 7h |
| WCC  | -3 → +3 | Rasio wick vs body candle terakhir |
| SRST | -4 → +3 | Kedekatan ke zona S/R aktif + strength |
| RSI  | -1 → +2 | RSI-14: >70=-1, 50-70=0, 30-50=+1, <30=+2 |
| MACD | -2 → +2 | Line vs Signal + Line vs Zero |
| MA   | -2 → +2 | Berapa MA (20/60/120/200) yang dilewati |
| IP   | -4 → +4 | MACD + Stoch pada TF daily/weekly/monthly |
| Tight| 0 → +2 | Jarak ke MA 3/5/10/20: VT=+2, T=+1 |

**Total Score** = Σ (score × weight) per fitur. Weight default = 1.0, di-update ML setiap `/scan`.

---

## ML Training

- **Model**: XGBoost multi-class classifier
- **Label**: `close[+3 candle] / close[0] - 1`
  - ≥ +0.5% → **NAIK** (1)
  - ≤ -0.5% → **TURUN** (-1)
  - sisanya → **NETRAL** (0)
- **Features**: 10 indikator + weighted total
- **Split**: 70% train / 30% test (time-ordered, tidak di-shuffle)
- **Output**: feature importance → weight baru per fitur
- **Warmup**: 200 candle pertama dipakai warmup, tidak di-train

---

## Catatan Penting

- ✅ Bot ini menggunakan **Binance Futures Testnet** — tidak ada uang nyata
- ✅ Selalu test dengan `/dryrun` sebelum `/scan` untuk pertama kali
- ✅ SL/TP dipasang sebagai order terpisah di Binance — akan ter-trigger otomatis
- ✅ Jika SL/TP sudah ter-trigger oleh Binance dan kamu coba `/close`, bot tetap aman (posisi sudah flat, bot hanya clean up state lokal)
- ⚠️ DeepSeek bisa timeout 30–90 detik — normal
- ⚠️ ML butuh minimal 200+30 bar di history sebelum bisa training
- ⚠️ Jangan jalankan `/scan` dua kali bersamaan untuk pair yang sama
