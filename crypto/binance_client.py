"""
binance_client.py — DEPRECATED SHIM

Semua fungsi sudah dipindah ke order/executor.py.
File ini hanya untuk backward compatibility — jangan tambah fungsi baru di sini.
Gunakan order.executor langsung untuk kode baru.
"""

import warnings
warnings.warn(
    "binance_client.py deprecated — gunakan order.executor langsung",
    DeprecationWarning,
    stacklevel=2,
)

from order.executor import (
    get_available_balance,
    get_symbol_info,
    get_mark_price,
    get_position_risk,
    get_open_positions,
    cancel_order,
    cancel_all_open_orders,
    get_order_status,
)

# Alias agar kode lama yang import nama lama tetap jalan
get_account_balance = get_available_balance
get_order_detail    = get_order_status
