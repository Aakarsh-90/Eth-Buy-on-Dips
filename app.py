# app.py — Buy-on-Dips Multi-Asset Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta

# ----------------------------------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Buy-the-Dip (Multi-Asset)", layout="wide")

# ----------------------------------------------------
# ASSET CONFIG
# ----------------------------------------------------
ASSETS = {
    "ETH-USD": {"label": "Ethereum (ETH-USD)", "kind": "crypto"},
    "GLD": {"label": "SPDR Gold Shares (GLD)", "kind": "etf"},
    "^SOX": {"label": "PHLX Semiconductor Index (^SOX)", "kind": "index"},
    "ARTY": {"label": "iShares Future AI & Tech (ARTY)", "kind": "etf"},
}

DEFAULT_START_DATE = date(2020, 9, 27)
DEFAULT_FIXED_END_DATE = date(2025, 8, 27)

# ----------------------------------------------------
# GLOBAL SETTINGS
# ----------------------------------------------------
DEFAULT_THRESHOLD_PCT = 5.0
DEFAULT_WINDOW_DAYS = 7
DEFAULT_BUY_AMOUNT = 150.0
DEFAULT_START_VALUE = 2500.0
DEFAULT_REF_PRICE = 354.31
DEFAULT_FEE_PCT = 0.10
DEFAULT_SLIPPAGE_PCT = 0.05
DEFAULT_COOLDOWN_DAYS = 0
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULT = 2.0
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_MAX = 45.0
DEFAULT_REQUIRE_NEW_HIGH_RESET = False
DEFAULT_MAX_SIG_PER_MONTH = 0
DEFAULT_TP_USE = False
DEFAULT_TP_TRIGGER_PCT = 20.0
DEFAULT_TP_SELL_PCT = 10.0
DEFAULT_TP_COOLDOWN_DAYS = 7
DEFAULT_MAX_INVESTED_USD = 0.0
DEFAULT_MAX_POSITION_VALUE_USD = 0.0

if "last_loader_error" not in st.session_state:
    st.session_state.last_loader_error = ""

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def _std_headers():
    return {
        "User-Agent": "Mozilla/5.0 (MultiAssetDipApp)",
        "Accept": "application/json",
    }

def _finalize_ohlc(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    df = df[~df.index.duplicated(keep="last")]
    keep = [c for c in ["High", "Low", "Close"] if c in df.columns]
    return df[keep].dropna(how="any")

def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in c if x]) for c in df.columns]
    return df

def format_usd(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

# ----------------------------------------------------
# SPOT PRICE HELPERS
# ----------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_spot_eth_usd():
    import requests
    try:
        r = requests.get("https://api.coinbase.com/v2/prices/ETH-USD/spot", timeout=15)
        return float(r.json()["data"]["amount"]), "Coinbase"
    except Exception:
        return np.nan, "—"

@st.cache_data(show_spinner=False, ttl=60)
def fetch_spot_yahoo_close(ticker: str):
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        return float(df["Close"].dropna().iloc[-1]), "Yahoo"
    except Exception:
        return np.nan, "—"

# ----------------------------------------------------
# OHLC LOADERS (ETH multi-source; others via Yahoo)
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_ohlc_yahoo_generic(ticker: str, start: date, end: date) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, start=start, end=end + timedelta(days=1),
            interval="1d", progress=False, auto_adjust=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = _flatten_multiindex_columns(df)
        cols = [c.lower() for c in df.columns]
        out = pd.DataFrame({
            "High": df.iloc[:, cols.index("high")] if "high" in cols else np.nan,
            "Low": df.iloc[:, cols.index("low")] if "low" in cols else np.nan,
            "Close": df.iloc[:, cols.index("close")] if "close" in cols else np.nan,
        })
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Yahoo generic fatal: {e}"
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_coinbase(start: date, end: date) -> pd.DataFrame:
    import requests, time
    try:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end + timedelta(days=1), tz="UTC")
        frames, cur = [], s
        while cur < e:
            chunk_end = min(cur + pd.Timedelta(days=200), e)
            r = requests.get(
                "https://api.exchange.coinbase.com/products/ETH-USD/candles",
                params={"granularity":86400, "start":cur.isoformat(), "end":chunk_end.isoformat()},
                timeout=30, headers=_std_headers())
            if r.status_code == 429: time.sleep(1)
            data = r.json()
            if not data: cur = chunk_end; continue
            df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
            df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
            df.set_index("ts", inplace=True)
            df = df[["high","low","close"]].astype(float)
            df.rename(columns={"high":"High","low":"Low","close":"Close"}, inplace=True)
            frames.append(df)
            cur = chunk_end; time.sleep(0.25)
        if not frames: return pd.DataFrame()
        return _finalize_ohlc(pd.concat(frames).sort_index(), start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Coinbase error: {e}"
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_binance(start: date, end: date) -> pd.DataFrame:
    import requests, time
    try:
        frames, cur = [], int(pd.Timestamp(start).timestamp()*1000)
        end_ts = int((pd.Timestamp(end)+pd.Timedelta(days=1)).timestamp()*1000)
        while cur < end_ts:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol":"ETHUSDT","interval":"1d","startTime":cur,"endTime":end_ts,"limit":1000},
                             timeout=30, headers=_std_headers())
            r.raise_for_status()
            klines = r.json()
            if not klines: break
            df = pd.DataFrame(klines, columns=["open_time","open","high","low","close","vol","ct","qav","not","tb","tq","ig"])
            df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
            df.set_index("ts", inplace=True)
            df = df[["high","low","close"]].astype(float)
            df.rename(columns={"high":"High","low":"Low","close":"Close"}, inplace=True)
            frames.append(df)
            cur = int(klines[-1][0]) + 86400000
            time.sleep(0.25)
        if not frames: return pd.DataFrame()
        return _finalize_ohlc(pd.concat(frames).sort_index(), start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Binance error: {e}"
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_kraken(start: date, end: date) -> pd.DataFrame:
    import requests, time
    try:
        frames = []
        cur = int(pd.Timestamp(start).timestamp())
        end_sec = int(pd.Timestamp(end + timedelta(days=1)).timestamp())
        while cur < end_sec:
            r = requests.get("https://api.kraken.com/0/public/OHLC",
                             params={"pair": "ETHUSD", "interval": 1440, "since": cur},
                             timeout=30, headers=_std_headers())
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                break
            result = data.get("result", {})
            key = next((k for k in result if k != "last"), None)
            if not key: break
            rows = result[key]
            if not rows: break
            df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","vol","count"])
            df["ts"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True).dt.tz_convert(None)
            df.set_index("ts", inplace=True)
            df = df[["high","low","close"]].astype(float).rename(columns={"high":"High","low":"Low","close":"Close"})
            frames.append(df)
            cur = int(rows[-1][0]) + 86400
            time.sleep(0.25)
        if not frames: return pd.DataFrame()
        return _finalize_ohlc(pd.concat(frames).sort_index(), start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Kraken error: {e}"
        return pd.DataFrame()

def fetch_ohlc(start: date, end: date, source: str, ticker: str) -> pd.DataFrame:
    kind = ASSETS.get(ticker, {}).get("kind", "etf")
    if kind == "crypto" and ticker.upper() == "ETH-USD":
        order = {
            "Yahoo Finance": [fetch_ohlc_yahoo_generic, fetch_ohlc_coinbase, fetch_ohlc_kraken, fetch_ohlc_binance],
            "Coinbase": [fetch_ohlc_coinbase, fetch_ohlc_binance, fetch_ohlc_kraken],
            "Kraken": [fetch_ohlc_kraken, fetch_ohlc_coinbase, fetch_ohlc_binance],
            "Binance (ETH/USDT)": [fetch_ohlc_binance, fetch_ohlc_coinbase],
        }.get(source, [fetch_ohlc_coinbase, fetch_ohlc_kraken, fetch_ohlc_binance])
        for fn in order:
            df = fn(start, end)
            if not df.empty: return df
        return pd.DataFrame()
    else:
        return fetch_ohlc_yahoo_generic(ticker, start, end)

# ----------------------------------------------------
# SIMPLE INDICATOR
# ----------------------------------------------------
def compute_indicators(ohlc: pd.DataFrame, window_days: int = 7):
    c = ohlc["Close"]
    h = ohlc["High"]
    l = ohlc["Low"]
    roll_max = c.rolling(window=window_days).max()
    dd = (roll_max - c)/roll_max * 100
    return pd.DataFrame({"Close": c, "High": h, "Low": l, "Drawdown": dd})

# ----------------------------------------------------
# STREAMLIT UI (TRUNCATED FOR DEMO)
# ----------------------------------------------------
st.title("📉 Multi-Asset Buy-the-Dip Dashboard (with Kraken Fix)")
st.write("✅ ETH, GLD, ^SOX, ARTY — with working Yahoo + Kraken loader.")

# Sample test block
start_date = date(2023, 1, 1)
end_date = date.today()

ticker = st.selectbox("Select asset", list(ASSETS.keys()), index=0)
data_source = st.selectbox("Data source", ["Yahoo Finance", "Coinbase", "Kraken", "Binance (ETH/USDT)"], index=0)

df = fetch_ohlc(start_date, end_date, data_source, ticker)
if df.empty:
    st.error("No data found for this selection.")
else:
    st.success(f"{ticker}: {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
    st.line_chart(df["Close"])

