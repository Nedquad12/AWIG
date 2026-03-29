# =============================================================
# config.py - Konfigurasi API Keys & Settings
# =============================================================
# CARA SETUP:
# 1. Daftar/login ke https://testnet.binancefuture.com
# 2. Generate API Key di Settings > API Management
# 3. Buat bot Telegram via @BotFather, copy token
# 4. Get Telegram Chat ID via @userinfobot
# =============================================================

# --- Binance Futures DEMO (Testnet) ---
BINANCE_API_KEY    = "NSWsgBKMo7UOd5fWk4P3CvL6sgq1nZKzzKj6SH2trdYxSzvfb45awxg0NPdOAcB1"
BINANCE_API_SECRET = "IifUKPtVyMunX2mdPlpkqXcwHUGOedvBgNlS9ErDS7WZwDJ0IwEPCEtshoPG3bge"

# Base URL testnet USDⓈ-M Futures
BINANCE_BASE_URL   = "https://demo-fapi.binance.com"

# --- Telegram Bot ---
TELEGRAM_BOT_TOKEN = "8117467835:AAFRni1PY5lmy9QC9hoh8YBjZi8RXorXCQQ"

# Chat ID yang diizinkan akses bot (whitelist).
# Bisa satu user atau list beberapa user.
ALLOWED_CHAT_IDS   = [6208519947,5751902978]  

# --- DeepSeek API ---
DEEPSEEK_API_KEY  = "ISI_DEEPSEEK_API_KEY_KAMU"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-reasoner"   # R1

# --- ML & Analisis ---
DEFAULT_INTERVAL   = "30m"    # timeframe default
CANDLE_LIMIT       = 1000     # candle untuk training
MIN_CANDLE_TRAIN   = 500      # minimal candle agar layak training
LOOKAHEAD          = 3        # prediksi N candle ke depan
LABEL_UP_PCT       = 0.005    # +0.5% → label naik
LABEL_DOWN_PCT     = -0.005   # -0.5% → label turun
CONFIDENCE_MIN     = 0.55     # minimal confidence untuk lanjut ke AI

# --- Order Execution ---
DEFAULT_LEVERAGE   = 10       # leverage default
RISK_PER_TRADE_PCT = 1.0      # % dari available balance per trade
WEIGHTS_DIR        = "weights" # folder simpan bobot ML per ticker

# --- Settings Umum ---
RECV_WINDOW = 5000   # milliseconds, rekomendasi ≤ 5000
