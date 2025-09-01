# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import numpy_financial as npf
import matplotlib.pyplot as plt
from datetime import date, timedelta

st.set_page_config(page_title="ETH Buy-the-Dip (5% in a Week)", layout="wide")

# --------------------------
# Defaults
# --------------------------
DEFAULT_START_DATE = date(2020, 9, 27)
DEFAULT_FIXED_END_DATE = date(2025, 8, 27)

DEFAULT_THRESHOLD_PCT = 5.0
DEFAULT_WINDOW_DAYS = 7
DEFAULT_BUY_AMOUNT = 150.0
DEFAULT_START_VALUE = 2500.00
DEFAULT_REF_PRICE   = 354.31

DEFAULT_FEE_PCT = 0.10
DEFAULT_SLIPPAGE_PCT = 0.05
DEFAULT_COOLDOWN_DAYS = 0

DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULT   = 2.0
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_MAX    = 45.0
DEFAULT_REQUIRE_NEW_HIGH_RESET = False
DEFAULT_MAX_SIG_PER_MONTH = 0

DEFAULT_TP_USE = False
DEFAULT_TP_BASIS = "Average cost"
DEFAULT_TP_TRIGGER_PCT = 20.0
DEFAULT_TP_SELL_PCT = 10.0
DEFAULT_TP_COOLDOWN_DAYS = 7

DEFAULT_MAX_INVESTED_USD = 0.0
DEFAULT_MAX_POSITION_VALUE_USD = 0.0

if "last_loader_error" not in st.session_state:
    st.session_state.last_loader_error = ""

# --------------------------
# Helpers
# --------------------------
def _std_headers():
    return {"User-Agent": "Mozilla/5.0 (compatible; ETHDipApp/1.0; +https://streamlit.io)"}

def _finalize_ohlc(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    df = df[~df.index.duplicated(keep="last")]
    keep = [c for c in ["High", "Low", "Close"] if c in df.columns]
    return df[keep].dropna(how="any")

# --- Live ETH Spot Price ---
@st.cache_data(show_spinner=False, ttl=60)
def fetch_spot_eth_usd():
    import requests
    try:
        r = requests.get("https://api.coinbase.com/v2/prices/ETH-USD/spot", timeout=15, headers=_std_headers())
        r.raise_for_status()
        return float(r.json()["data"]["amount"]), "Coinbase"
    except: pass
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol":"ETHUSDT"}, timeout=15, headers=_std_headers())
        r.raise_for_status()
        return float(r.json()["price"]), "Binance"
    except: pass
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price", params={"ids":"ethereum","vs_currencies":"usd"}, timeout=15, headers=_std_headers())
        r.raise_for_status()
        return float(r.json()["ethereum"]["usd"]), "CoinGecko"
    except: return np.nan, "—"

# --------------------------
# (… all your fetch_ohlc_* functions here unchanged …)
# --------------------------
# [to save space, keep your Yahoo/Coingecko/Binance/Coinbase/Kraken functions as-is]

# --------------------------
# Indicators & Strategy
# --------------------------
# (keep compute_indicators, compute_base_signals, robust_xirr, simulate_strategy as-is)

def format_usd(x):
    try: return f"${x:,.2f}"
    except: return str(x)

def price_on_or_near(ind_df: pd.DataFrame, anchor: date) -> float:
    if ind_df.empty: return float("nan")
    ts = pd.Timestamp(anchor)
    if ts in ind_df.index: return float(ind_df.loc[ts, "Close"])
    pos = ind_df.index.get_indexer([ts], method="nearest")[0]
    return float(ind_df.iloc[pos]["Close"])

# --------------------------
# UI
# --------------------------
st.title("📉 ETH Buy-on-Dips — with Risk Controls")
st.caption("Educational tool. Not financial advice.")

# --- Sidebar (keep your existing inputs here) ---

# --- Fetch OHLC and indicators (keep your existing logic) ---

# --- Simulate DIP strategy (keep existing logic) ---

# --------------------------
# Output (Dip strategy)
# --------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades executed", f"{summary['n_trades']}")
m2.metric("ETH held (final)", f"{summary['final_eth']:.6f}")
m3.metric("Ending value", format_usd(summary["terminal_value"]))
m4.metric("XIRR", f"{summary['irr_xirr']*100:.2f}%" if pd.notna(summary["irr_xirr"]) else "N/A")

avg_basis_now = details["ind"]["AvgCostBasis"].iloc[-1]
a1, a2 = st.columns(2)
a1.metric("Avg cost (current)", format_usd(avg_basis_now))
spot_price, spot_src = fetch_spot_eth_usd()
a2.metric("ETH price (now)", f"{format_usd(spot_price)} · {spot_src}")

i1, i2 = st.columns(2)
i1.metric("Total invested", format_usd(summary["total_invested"]))
i2.metric("P/L vs invested", f"{format_usd(summary['absolute_pnl'])} ({summary['pct_return_on_invested']:.2f}%)")

# --- Charts & Tables (keep same) ---

with left:
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        st.download_button("Download trades (CSV)", trades_df.to_csv(index=False), "trades.csv", "text/csv", key="dl_trades_main")

with right:
    cf = details["cashflows"]
    st.dataframe(cf, use_container_width=True, hide_index=True)
    st.download_button("Download cash flows (CSV)", cf.to_csv(index=False), "cashflows.csv", "text/csv", key="dl_cf_main")

# --------------------------
# Benchmarks
# --------------------------
st.header("📊 Benchmarks")

def render_block(sim_summary, sim_details, label_prefix=""):
    slug = "".join(ch if ch.isalnum() else "_" for ch in label_prefix).strip("_")

    # metrics...
    st.download_button("Download trades (CSV)",
        sim_details["trades_df"].to_csv(index=False),
        "trades.csv","text/csv", key=f"dl_trades_{slug}")

    st.download_button("Download cash flows (CSV)",
        sim_details["cashflows"].to_csv(index=False),
        "cashflows.csv","text/csv", key=f"dl_cf_{slug}")

# (keep Buy & Hold and Monthly DCA tab logic the same, but when calling render_block give label_prefix "bh" and "dca")

