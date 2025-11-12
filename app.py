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
    "GLD":     {"label": "SPDR Gold Shares (GLD)", "kind": "etf"},
    "^SOX":    {"label": "PHLX Semiconductor Index (^SOX)", "kind": "index"},
    "ARTY":    {"label": "iShares Future AI & Tech (ARTY)", "kind": "etf"},
}

DEFAULT_START_DATE = date(2020, 9, 27)
DEFAULT_FIXED_END_DATE = date(2025, 8, 27)

# ----------------------------------------------------
# GLOBAL DEFAULTS
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
    return df[["High","Low","Close"]].dropna(how="any")

def _flatten_multiindex_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = [str(x) for x in tup if x not in (None, "", " ")]
            new_cols.append("_".join(parts) if parts else "col")
        df.columns = new_cols
    return df

def format_usd(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

# ----------------------------------------------------
# DATA LOADERS
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_ohlc_yahoo_generic(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end + timedelta(days=1),
                         interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = _flatten_multiindex_columns(df)
        def pick_col(name):
            if name in df.columns:
                return df[name]
            cands = [c for c in df.columns if c.split("_")[-1]==name or c.split(" ")[-1]==name]
            return df[cands[0]] if cands else pd.Series(index=df.index, dtype="float64")
        out = pd.DataFrame({
            "High": pd.to_numeric(pick_col("High"), errors="coerce"),
            "Low":  pd.to_numeric(pick_col("Low"),  errors="coerce"),
            "Close":pd.to_numeric(pick_col("Close"),errors="coerce"),
        })
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error=f"Yahoo fatal: {e}"
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_coinbase(start, end):
    import requests, time
    try:
        s=pd.Timestamp(start,tz="UTC"); e=pd.Timestamp(end+timedelta(days=1),tz="UTC")
        frames=[]; cur=s
        while cur<e:
            chunk=min(cur+pd.Timedelta(days=200),e)
            r=requests.get("https://api.exchange.coinbase.com/products/ETH-USD/candles",
                           params={"granularity":86400,"start":cur.isoformat(),"end":chunk.isoformat()},
                           timeout=30,headers=_std_headers())
            r.raise_for_status()
            data=r.json()
            if not data: cur=chunk; continue
            df=pd.DataFrame(data,columns=["time","low","high","open","close","vol"])
            df["ts"]=pd.to_datetime(df["time"],unit="s",utc=True).dt.tz_convert(None)
            df.set_index("ts",inplace=True)
            df=df[["high","low","close"]].astype(float)
            df.rename(columns={"high":"High","low":"Low","close":"Close"},inplace=True)
            frames.append(df); cur=chunk; time.sleep(0.25)
        if not frames: return pd.DataFrame()
        return _finalize_ohlc(pd.concat(frames).sort_index(),start,end)
    except Exception as e:
        st.session_state.last_loader_error=f"Coinbase: {e}"; return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_binance(start,end):
    import requests,time
    try:
        s=int(pd.Timestamp(start).timestamp()*1000); e=int((pd.Timestamp(end)+pd.Timedelta(days=1)).timestamp()*1000)
        cur=s; frames=[]
        while cur<e:
            r=requests.get("https://api.binance.com/api/v3/klines",
                           params={"symbol":"ETHUSDT","interval":"1d","startTime":cur,"endTime":e,"limit":1000},
                           timeout=30,headers=_std_headers())
            r.raise_for_status(); kl=r.json()
            if not kl: break
            df=pd.DataFrame(kl,columns=["t","o","h","l","c","v","ct","q","n","tb","tq","ig"])
            df["ts"]=pd.to_datetime(df["t"],unit="ms",utc=True).dt.tz_convert(None)
            df.set_index("ts",inplace=True)
            df=df[["h","l","c"]].astype(float)
            df.rename(columns={"h":"High","l":"Low","c":"Close"},inplace=True)
            frames.append(df); cur=int(kl[-1][0])+86400000; time.sleep(0.25)
        if not frames: return pd.DataFrame()
        return _finalize_ohlc(pd.concat(frames).sort_index(),start,end)
    except Exception as e:
        st.session_state.last_loader_error=f"Binance: {e}"; return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_kraken(start,end):
    import requests,time
    try:
        frames=[]; cur=int(pd.Timestamp(start).timestamp()); end_s=int(pd.Timestamp(end+timedelta(days=1)).timestamp())
        while cur<end_s:
            r=requests.get("https://api.kraken.com/0/public/OHLC",
                           params={"pair":"ETHUSD","interval":1440,"since":cur},
                           timeout=30,headers=_std_headers())
            r.raise_for_status(); data=r.json()
            if data.get("error"): break
            result=data.get("result",{}); key=next((k for k in result if k!="last"),None)
            if not key: break
            rows=result[key]
            if not rows: break
            df=pd.DataFrame(rows,columns=["time","open","high","low","close","vwap","vol","count"])
            df["ts"]=pd.to_datetime(df["time"].astype(int),unit="s",utc=True).dt.tz_convert(None)
            df.set_index("ts",inplace=True)
            df=df[["high","low","close"]].astype(float)
            df.rename(columns={"high":"High","low":"Low","close":"Close"},inplace=True)
            frames.append(df); cur=int(rows[-1][0])+86400; time.sleep(0.25)
        if not frames: return pd.DataFrame()
        return _finalize_ohlc(pd.concat(frames).sort_index(),start,end)
    except Exception as e:
        st.session_state.last_loader_error=f"Kraken: {e}"; return pd.DataFrame()

def fetch_ohlc(start,end,source,ticker):
    kind=ASSETS.get(ticker,{}).get("kind","etf")
    if kind=="crypto" and ticker.upper()=="ETH-USD":
        order={
            "Yahoo Finance":[(lambda s,e:fetch_ohlc_yahoo_generic("ETH-USD",s,e)),fetch_ohlc_coinbase,fetch_ohlc_kraken,fetch_ohlc_binance],
            "Coinbase":[fetch_ohlc_coinbase,fetch_ohlc_binance,fetch_ohlc_kraken,(lambda s,e:fetch_ohlc_yahoo_generic("ETH-USD",s,e))],
            "Kraken":[fetch_ohlc_kraken,fetch_ohlc_coinbase,fetch_ohlc_binance,(lambda s,e:fetch_ohlc_yahoo_generic("ETH-USD",s,e))],
            "Binance (ETH/USDT)":[fetch_ohlc_binance,fetch_ohlc_coinbase,fetch_ohlc_kraken,(lambda s,e:fetch_ohlc_yahoo_generic("ETH-USD",s,e))]
        }.get(source,[fetch_ohlc_coinbase,fetch_ohlc_kraken,fetch_ohlc_binance,(lambda s,e:fetch_ohlc_yahoo_generic("ETH-USD",s,e))])
        for fn in order:
            df=fn(start,end)
            if not df.empty: return df
        return pd.DataFrame()
    else:
        return fetch_ohlc_yahoo_generic(ticker,start,end)

# ----------------------------------------------------
# SIMPLE INDICATOR / DEMO VIEW
# ----------------------------------------------------
st.title("📉 Multi-Asset Buy-on-Dips Dashboard (Stable Build)")
st.sidebar.header("Parameters")
start_date=st.sidebar.date_input("Start Date",value=DEFAULT_START_DATE)
end_date=st.sidebar.date_input("End Date",value=date.today())
ticker=st.sidebar.selectbox("Select Asset",list(ASSETS.keys()),index=0)
data_source=st.sidebar.selectbox("Data Source",["Yahoo Finance","Coinbase","Kraken","Binance (ETH/USDT)"],index=0)

if st.sidebar.button("Clear Cache & Reload"):
    st.cache_data.clear(); st.rerun()

ohlc=fetch_ohlc(start_date,end_date,data_source,ticker)
if ohlc.empty:
    st.error("No data found for this selection.")
    st.stop()

st.success(f"{ticker}: {len(ohlc)} rows from {ohlc.index.min().date()} to {ohlc.index.max().date()}")

fig,ax=plt.subplots()
ax.plot(ohlc.index,ohlc["Close"],label="Close Price")
ax.set_title(f"{ASSETS[ticker]['label']} — Close Price")
ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
st.pyplot(fig,use_container_width=True)
st.dataframe(ohlc.tail(),use_container_width=True)
