"""Central configuration for the swing trading agent.

Loads environment variables from a local .env file and exposes them as simple
module-level constants. Secrets live in .env (git-ignored) and are never
hardcoded here.
"""

import os

from dotenv import load_dotenv

# Read key=value pairs from .env into the process environment.
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment, accepting common truthy strings."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Data and news APIs ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# --- Interactive Brokers (paper) ---
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

# --- Agent behavior ---
# Autonomy gate mode: signal_only | approve | semi_auto | full_auto. Start safe.
AUTONOMY_MODE = os.getenv("AUTONOMY_MODE", "approve")

# Fraction of account equity risked per trade (1% default).
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

# --- Signal generation ---
# Breakout entry: latest close must exceed the high of the prior N sessions.
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "20"))

# --- Risk and position sizing ---
# ATR (Average True Range) settings for volatility-based stops.
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLE = float(os.getenv("ATR_MULTIPLE", "2.0"))
# Cap any single position at this fraction of account equity.
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10"))

# --- Portfolio-level risk limits ---
# Cap all names within one sector at this fraction of equity.
MAX_SECTOR_PCT = float(os.getenv("MAX_SECTOR_PCT", "0.30"))
# Cap total deployed capital at this fraction of equity.
MAX_TOTAL_EXPOSURE = float(os.getenv("MAX_TOTAL_EXPOSURE", "0.60"))

# --- Execution safety ---
# When True, the decision runner prints proposals but places no orders.
DRY_RUN = _env_bool("DRY_RUN", False)
