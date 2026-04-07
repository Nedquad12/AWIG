"""
binance_client.py — DEPRECATED SHIM

Semua fungsi sudah dipindah ke order/executor.py.
File ini hanya untuk backward compatibility agar telegram_bot.py
dan kode lain yang import `binance_client as bc` tetap berjalan tanpa perubahan.

Jangan tambah fungsi baru di sini. Gunakan order.executor langsung.
"""

import warnings
warnings.warn(
    "binance_client.py deprecated — gunakan order.executor langsung",
    DeprecationWarning,
    stacklevel=2,
)

from order.executor import (
    # Public endpoints
    get_server_time,
    get_ticker_price,
    get_24hr_ticker,

    # Account
    get_account_info,
    get_account_balance_full  as get_account_balance,  # alias nama lama
    get_available_balance,

    # Positions
    get_position_risk,
    get_open_positions,

    # Orders
    get_open_orders,
    get_all_orders,
    get_order_status          as get_order_detail,      # alias nama lama
    get_order_status,
    cancel_order,
    cancel_all_open_orders,

    # Income
    get_income_history,

    # Market data
    get_symbol_info,
    get_mark_price,
)
