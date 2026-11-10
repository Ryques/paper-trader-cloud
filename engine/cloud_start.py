import os
import sys

required = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
missing = [name for name in required if not os.environ.get(name, "").strip()]
if missing:
    sys.exit("Missing paper credentials: " + ", ".join(missing))
mode = os.environ.get("PAPER_TRADER_MODE", "preview")
args = {"preview": [], "trade": ["--submit", "--watch"]}
if mode not in args:
    sys.exit("Set PAPER_TRADER_MODE to preview or trade.")
os.execv(sys.executable, [sys.executable, "stock_learning_trader.py", *args[mode]])
