# app.py — Buy-on-Dips Multi-Asset Dashboard (robust ticker fallbacks)
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
# ASSET CONFIG (with ticker fallback lists)
# ----------------------------------------------------
ASSETS = {
    "ETH-USD": {
        "label": "Ethereum (ETH-USD)",
        "kind": "crypto",
        "symbols": ["ETH-USD"],  # crypto path ignores this list (uses exchanges), but keep for uniformity
    },
    "GLD": {
        "label": "SPDR Gold Shares (GLD)",
        "kind": "etf",
        "symbols": ["GLD", "IAU"],
    },
    "^SOX": {
        "label": "PHLX Semiconductor Index (^SOX)",
        "kind": "index",
        "symbols": ["^SOX", "SOXX"],  # SOXX ETF proxy fallback
    },
    "ARTY": {
        "label": "iShares Future AI & Tech (ARTY)",
        "kind": "etf",
        "symbols": ["ARTY", "CTRU", "BOTZ", "IRBO"],  # robust fallbacks if ARTY has no Yahoo coverage
    },
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

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def _std_headers():
    return {
        "User-Agent": "Mozilla/5.0 (MultiAssetDipApp)",
        "Accept": "application/json",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
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

def price_on_or_near(ind_df: pd.DataFrame, anchor: date) -> float:
    if ind_df is None or ind_df.empty:
        return float("nan")
    ts = pd.Timestamp(anchor)
    if ts in ind_df.index:
        return float(ind_df.loc[ts, "Close"])
    pos = ind_df.index.get_indexer([ts], method="nearest")[0]
    return float(ind_df.iloc[pos]["Close"])

# ----------------------------------------------------
# SPOT PRICE HELPERS
# ----------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_spot_eth_usd():
    import requests
    try:
        r = requests.get("https://api.coinbase.com/v2/prices/ETH-USD/spot", timeout=15, headers=_std_headers())
        r.raise_for_status()
        return float(r.json()["data"]["amount"]), "Coinbase"
    except Exception:
        try:
            r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                             params={"ids": "ethereum", "vs_currencies": "usd"},
                             timeout=15, headers=_std_headers())
            r.raise_for_status()
            return float(r.json()["ethereum"]["usd"]), "CoinGecko"
        except Exception:
            return np.nan, "—"

@st.cache_data(show_spinner=False, ttl=60)
def fetch_spot_yahoo_close(ticker: str):
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return np.nan, "—"
        return float(df["Close"].dropna().iloc[-1]), "Yahoo"
    except Exception:
        return np.nan, "—"

# ----------------------------------------------------
# OHLC LOADERS (ETH multi-source; others via Yahoo w/ fallbacks)
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_ohlc_yahoo_generic(symbol: str, start: date, end: date) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end + timedelta(days=1),  # yfinance end is exclusive
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = _flatten_multiindex_columns(df)

        def pick_col(name: str):
            if name in df.columns:
                return df[name]
            cands = [c for c in df.columns if c.split("_")[-1] == name or c.split(" ")[-1] == name]
            return df[cands[0]] if cands else pd.Series(index=df.index, dtype="float64")

        out = pd.DataFrame({
            "High":  pd.to_numeric(pick_col("High"),  errors="coerce"),
            "Low":   pd.to_numeric(pick_col("Low"),   errors="coerce"),
            "Close": pd.to_numeric(pick_col("Close"), errors="coerce"),
        })
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Yahoo generic fatal: {e!s}"
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_coinbase(start: date, end: date) -> pd.DataFrame:
    import requests, time
    try:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end + timedelta(days=1), tz="UTC")
        step = pd.Timedelta(days=200)
        frames, cur = [], s
        while cur < e:
            chunk_end = min(cur + step, e)
            r = requests.get("https://api.exchange.coinbase.com/products/ETH-USD/candles",
                             params={"granularity":86400, "start":cur.isoformat(), "end":chunk_end.isoformat()},
                             timeout=30, headers=_std_headers())
            if r.status_code == 429:
                time.sleep(1.0)
                r = requests.get("https://api.exchange.coinbase.com/products/ETH-USD/candles",
                                 params={"granularity":86400, "start":cur.isoformat(), "end":chunk_end.isoformat()},
                                 timeout=30, headers=_std_headers())
            r.raise_for_status()
            data = r.json()
            if not data:
                cur = chunk_end
                continue
            df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
            df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
            df.set_index("ts", inplace=True)
            df = df[["high","low","close"]].astype(float).rename(columns={"high":"High","low":"Low","close":"Close"})
            frames.append(df.sort_index())
            cur = chunk_end
            time.sleep(0.25)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Coinbase error: {e!s}"
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_binance(start: date, end: date) -> pd.DataFrame:
    import requests, time
    try:
        start_ts = int(pd.Timestamp(start).timestamp() * 1000)
        end_ts   = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000)
        frames, cur = [], start_ts
        while cur < end_ts:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol":"ETHUSDT","interval":"1d","startTime":cur,"endTime":end_ts,"limit":1000},
                             timeout=30, headers=_std_headers())
            r.raise_for_status()
            klines = r.json()
            if not klines:
                break
            df = pd.DataFrame(klines, columns=[
                "open_time","open","high","low","close","volume","close_time",
                "quote_asset_volume","number_of_trades","taker_buy_base",
                "taker_buy_quote","ignore"
            ])
            df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
            df.set_index("ts", inplace=True)
            df = df[["high","low","close"]].astype(float)
            df.rename(columns={"high":"High","low":"Low","close":"Close"}, inplace=True)
            frames.append(df)
            cur = int(klines[-1][0]) + 24*60*60*1000
            time.sleep(0.25)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Binance error: {e!s}"
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
            key = next((k for k in result.keys() if k != "last"), None)
            if not key:
                break
            rows = result[key]
            if not rows:
                break
            df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"])
            df["ts"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True).dt.tz_convert(None)
            df.set_index("ts", inplace=True)
            df = df[["high","low","close"]].astype(float).rename(columns={"high":"High","low":"Low","close":"Close"})
            frames.append(df)
            cur = int(rows[-1][0]) + 24*60*60
            time.sleep(0.25)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Kraken error: {e!s}"
        return pd.DataFrame()

def fetch_ohlc_with_symbol_fallbacks(start: date, end: date, symbols: list[str]):
    """Try multiple Yahoo symbols until one returns data; return (df, used_symbol)."""
    for sym in symbols:
        df = fetch_ohlc_yahoo_generic(sym, start, end)
        if df is not None and not df.empty:
            return df, sym
    return pd.DataFrame(), None

def fetch_ohlc(start: date, end: date, source: str, asset_key: str):
    """Unified fetch. For ETH-USD use exchange fallbacks; for others try Yahoo symbol list."""
    meta = ASSETS.get(asset_key, {})
    kind = meta.get("kind", "etf")
    if kind == "crypto" and asset_key.upper() == "ETH-USD":
        order = {
            "Yahoo Finance": [
                (lambda s, e: fetch_ohlc_yahoo_generic("ETH-USD", s, e)),
                fetch_ohlc_coinbase,
                fetch_ohlc_kraken,
                fetch_ohlc_binance,
            ],
            "Coinbase": [
                fetch_ohlc_coinbase,
                fetch_ohlc_binance,
                fetch_ohlc_kraken,
                (lambda s, e: fetch_ohlc_yahoo_generic("ETH-USD", s, e)),
            ],
            "Kraken": [
                fetch_ohlc_kraken,
                fetch_ohlc_coinbase,
                fetch_ohlc_binance,
                (lambda s, e: fetch_ohlc_yahoo_generic("ETH-USD", s, e)),
            ],
            "Binance (ETH/USDT)": [
                fetch_ohlc_binance,
                fetch_ohlc_coinbase,
                fetch_ohlc_kraken,
                (lambda s, e: fetch_ohlc_yahoo_generic("ETH-USD", s, e)),
            ],
            "CoinGecko": [
                (lambda s, e: fetch_ohlc_yahoo_generic("ETH-USD", s, e)),
            ],
        }.get(source, [
            fetch_ohlc_coinbase,
            fetch_ohlc_kraken,
            fetch_ohlc_binance,
            (lambda s, e: fetch_ohlc_yahoo_generic("ETH-USD", s, e)),
        ])
        for fn in order:
            df = fn(start, end)
            if df is not None and not df.empty:
                return df, "ETH-USD"
        return pd.DataFrame(), None
    else:
        symbols = meta.get("symbols", [asset_key])
        return fetch_ohlc_with_symbol_fallbacks(start, end, symbols)

# ----------------------------------------------------
# INDICATORS & SIGNALS
# ----------------------------------------------------
def compute_indicators(ohlc: pd.DataFrame, window_days: int, rsi_period: int, atr_period: int):
    if ohlc is None or ohlc.empty:
        return pd.DataFrame()

    close = pd.to_numeric(ohlc["Close"], errors="coerce")
    high  = pd.to_numeric(ohlc["High"],  errors="coerce")
    low   = pd.to_numeric(ohlc["Low"],   errors="coerce")

    roll_max = close.rolling(window=window_days, min_periods=1).max()

    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/float(rsi_period), adjust=False).mean()
    roll_down = down.ewm(alpha=1/float(rsi_period), adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/float(atr_period), adjust=False).mean()

    drawdown_pct = (roll_max - close) / roll_max * 100.0
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr_pct = (atr / close) * 100.0

    out = pd.DataFrame({
        "Close": close,
        "High": high,
        "Low": low,
        "RollingMax": roll_max,
        "DrawdownPct": drawdown_pct,
        "SMA50": sma50,
        "SMA200": sma200,
        "RSI": rsi,
        "ATR": atr,
        "ATR_Pct": atr_pct,
    }).dropna(subset=["Close"])
    out = out[~out.index.duplicated(keep="last")]
    return out

def compute_base_signals(ind: pd.DataFrame, threshold_mode: str, fixed_pct: float, atr_mult: float) -> pd.Series:
    thr_series = ind["ATR_Pct"] * atr_mult if threshold_mode.startswith("ATR") else pd.Series(fixed_pct, index=ind.index)
    cond = ind["DrawdownPct"] >= thr_series
    return cond & (~cond.shift(1).fillna(False))

def robust_xirr(dates, amounts):
    if len(dates) != len(amounts) or len(dates) == 0:
        return np.nan
    dfcf = pd.DataFrame({"date": pd.to_datetime(dates).date, "amt": amounts})
    dfcf = dfcf.groupby("date", as_index=False)["amt"].sum().sort_values("date")
    amts = dfcf["amt"].to_numpy(dtype=float)
    dts  = pd.to_datetime(dfcf["date"])
    if not (np.any(amts > 0) and np.any(amts < 0)):
        return np.nan
    t0 = dts.iloc[0]
    years = (dts - t0).dt.days.to_numpy(dtype=float) / 365.0
    def npv(r): return np.sum(amts / (1.0 + r) ** years)
    lo, hi = -0.999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    tries = 0
    while f_lo * f_hi > 0 and hi < 1e6 and tries < 60:
        hi *= 1.5; f_hi = npv(hi); tries += 1
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-10:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0

# ----------------------------------------------------
# SIMULATION
# ----------------------------------------------------
def simulate_strategy(
    ind: pd.DataFrame,
    base_signal: pd.Series,
    buy_amount: float,
    start_value_usd: float,
    ref_price_usd: float,
    fee_pct: float = 0.0,
    slippage_pct: float = 0.0,
    cooldown_days: int = 0,
    trend_filter: str = "None",
    use_rsi: bool = False,
    rsi_max: float = 45.0,
    require_new_high_reset: bool = False,
    max_signals_per_month: int = 0,
    tp_use: bool = False,
    tp_basis: str = "Average cost",
    tp_trigger_pct: float = 20.0,
    tp_sell_pct: float = 10.0,
    tp_cooldown_days: int = 7,
    max_invested_usd: float = 0.0,
    max_position_value_usd: float = 0.0
):
    idx = ind.index
    if not isinstance(base_signal, pd.Series):
        base_signal = pd.Series(base_signal, index=idx)
    signal = base_signal.reindex(idx, fill_value=False).astype(bool).to_numpy()

    close    = ind["Close"].to_numpy()
    roll_max = ind["RollingMax"].to_numpy()
    sma50    = ind["SMA50"].to_numpy()
    sma200   = ind["SMA200"].to_numpy()
    rsi      = ind["RSI"].to_numpy()
    drawdown = ind["DrawdownPct"].to_numpy()

    current_units = start_value_usd / ref_price_usd if ref_price_usd > 0 else 0.0
    total_cost_excl_fees = start_value_usd
    last_buy_price = ref_price_usd
    last_buy_date = None
    last_tp_date  = None
    last_signal_rollmax_at_buy = None
    month_counts = {}

    cashflows_dates = [idx[0].date()]
    cashflows_amounts = [-float(start_value_usd)]
    trades = []

    units_series = pd.Series(index=idx, dtype=float)
    cost_basis_series = pd.Series(index=idx, dtype=float)

    def avg_cost_per_unit():
        return total_cost_excl_fees / current_units if current_units > 0 else np.nan

    for i, dt in enumerate(idx):
        # take-profit
        if tp_use and current_units > 0:
            can_tp = (last_tp_date is None) or ((dt.date() - last_tp_date).days >= int(tp_cooldown_days))
            if can_tp:
                basis_price = avg_cost_per_unit() if tp_basis == "Average cost" else last_buy_price
                if not pd.isna(basis_price):
                    trigger_price = basis_price * (1.0 + float(tp_trigger_pct) / 100.0)
                    if close[i] >= trigger_price:
                        sell_units = max(0.0, current_units * (float(tp_sell_pct) / 100.0))
                        if sell_units > 0:
                            exec_price = float(close[i]) * (1.0 - float(slippage_pct) / 100.0)
                            gross = sell_units * exec_price
                            fee   = gross * (float(fee_pct) / 100.0)
                            net   = gross - fee
                            cost_removed = (avg_cost_per_unit() * sell_units) if not pd.isna(avg_cost_per_unit()) else 0.0
                            total_cost_excl_fees -= cost_removed
                            current_units -= sell_units
                            cashflows_dates.append(dt.date())
                            cashflows_amounts.append(net)
                            trades.append({
                                "Type":"SELL","Date":dt.date(),"Price_Close":float(close[i]),
                                "Executed_Price":exec_price,"Units":-sell_units,"Gross_Proceeds":gross,
                                "Fee_%":float(fee_pct),"Fee_Cash":fee,"Net_Proceeds":net,
                                "Basis_Type":tp_basis,"Basis_Price_Used":float(basis_price),
                            })
                            last_tp_date = dt.date()

        # buy logic
        execute_buy = False
        if signal[i]:
            ym = (dt.year, dt.month)
            if (max_signals_per_month > 0) and (month_counts.get(ym, 0) >= max_signals_per_month):
                execute_buy = False
            else:
                if (last_buy_date is None) or ((dt.date() - last_buy_date).days >= int(cooldown_days)):
                    if require_new_high_reset and (last_signal_rollmax_at_buy is not None):
                        execute_buy = bool(roll_max[i] > last_signal_rollmax_at_buy)
                    else:
                        execute_buy = True

        if execute_buy and trend_filter == "Close > 200D SMA":
            execute_buy = bool(close[i] > sma200[i])
        elif execute_buy and trend_filter == "50D SMA > 200D SMA":
            execute_buy = bool(sma50[i] > sma200[i])

        if execute_buy and use_rsi:
            rsi_val = rsi[i]
            if pd.isna(rsi_val) or not (rsi_val <= float(rsi_max)):
                execute_buy = False

        if execute_buy:
            total_invested_so_far = -sum(a for a in cashflows_amounts if a < 0)
            prospective_cash_out = float(buy_amount) * (1.0 + float(fee_pct) / 100.0)
            if (max_invested_usd > 0) and (total_invested_so_far + prospective_cash_out > float(max_invested_usd)):
                execute_buy = False
            position_value_now = current_units * float(close[i])
            if (max_position_value_usd > 0) and (position_value_now > float(max_position_value_usd)):
                execute_buy = False

        if execute_buy:
            exec_price = float(close[i]) * (1.0 + float(slippage_pct) / 100.0)
            units_bought = float(buy_amount) / exec_price
            fee_cash = float(buy_amount) * (float(fee_pct) / 100.0)
            total_cash_out = float(buy_amount) + fee_cash

            current_units += units_bought
            total_cost_excl_fees += float(buy_amount)
            last_buy_price = exec_price
            last_buy_date  = dt.date()
            last_signal_rollmax_at_buy = roll_max[i]
            ym = (dt.year, dt.month)
            month_counts[ym] = month_counts.get(ym, 0) + 1

            cashflows_dates.append(dt.date())
            cashflows_amounts.append(-total_cash_out)
            trades.append({
                "Type":"BUY","Date":dt.date(),"Price_Close":float(close[i]),
                "Executed_Price":exec_price,"Units":units_bought,
                "USD_Spent_Excl_Fee":float(buy_amount),"Fee_%":float(fee_pct),"Fee_Cash":fee_cash,
                "Total_Cash_Out":total_cash_out,"Drawdown% (on signal)":float(drawdown[i]),
                "RSI": float(rsi[i]) if not pd.isna(rsi[i]) else np.nan,
                "SMA50": float(sma50[i]) if not pd.isna(sma50[i]) else np.nan,
                "SMA200": float(sma200[i]) if not pd.isna(sma200[i]) else np.nan,
            })

        units_series.iloc[i] = current_units
        cost_basis_series.iloc[i] = (total_cost_excl_fees / current_units) if current_units > 0 else np.nan

    ind = ind.copy()
    ind["Units"] = units_series
    ind["AvgCostBasis"] = cost_basis_series

    portfolio_value = ind["Units"] * ind["Close"]
    terminal_value = float(portfolio_value.iloc[-1])
    cashflows_dates.append(idx[-1].date())
    cashflows_amounts.append(terminal_value)

    irr = robust_xirr(pd.to_datetime(cashflows_dates), cashflows_amounts)
    total_invested = float(-sum(a for a in cashflows_amounts if a < 0))
    summary = {
        "final_units": float(ind["Units"].iloc[-1]),
        "terminal_value": float(terminal_value),
        "total_invested": total_invested,
        "n_trades": int(len(trades)),
        "irr_xirr": irr,
        "absolute_pnl": float(terminal_value - total_invested),
        "pct_return_on_invested": float((terminal_value / total_invested - 1.0) * 100.0) if total_invested > 0 else np.nan,
    }
    details = {
        "portfolio_value": portfolio_value,
        "trades_df": pd.DataFrame(trades),
        "cashflows": pd.DataFrame({"Date": cashflows_dates, "Amount": cashflows_amounts}),
        "ind": ind,
    }
    return summary, details

# ----------------------------------------------------
# UI — Controls
# ----------------------------------------------------
st.sidebar.header("Backtest Period")
start_date = st.sidebar.date_input("Start date", value=DEFAULT_START_DATE)
end_mode = st.sidebar.radio("End date mode", ["Rolling (use today's date)", "Fixed date"], index=0)
end_date = date.today() if end_mode.startswith("Rolling") else st.sidebar.date_input("Fixed end date", value=DEFAULT_FIXED_END_DATE, min_value=start_date)
if end_mode.startswith("Rolling"):
    st.sidebar.info(f"Using today's date: {end_date.isoformat()}")

st.sidebar.header("Asset (single-asset panel)")
asset_key = st.sidebar.selectbox("Choose asset", options=list(ASSETS.keys()), format_func=lambda k: ASSETS[k]["label"], index=0)

st.title(f"📉 {ASSETS[asset_key]['label']} — Buy-on-Dips with Risk Controls")

st.sidebar.header("Data source (ETH only uses fallbacks)")
data_source = st.sidebar.selectbox(
    "Data source",
    ["Coinbase", "Kraken", "Yahoo Finance", "Binance (ETH/USDT)", "CoinGecko"],
    index=0,
    help="For ETH-USD: multi-source fallback. For non-crypto: Yahoo with symbol fallbacks."
)
if st.sidebar.button("🔄 Clear data cache"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Signal Logic")
threshold_mode = st.sidebar.selectbox("Threshold mode", ["Fixed % (classic 5%)", "ATR multiple (vol-adaptive)"], index=0)
if threshold_mode.startswith("ATR"):
    atr_mult = st.sidebar.number_input("ATR multiple", min_value=0.5, max_value=5.0, value=DEFAULT_ATR_MULT, step=0.1)
    threshold_pct = DEFAULT_THRESHOLD_PCT
else:
    atr_mult = DEFAULT_ATR_MULT
    threshold_pct = st.sidebar.number_input("Dip threshold (%)", min_value=1.0, max_value=50.0, value=DEFAULT_THRESHOLD_PCT, step=0.5)
window_days = st.sidebar.number_input("Week window (days)", min_value=2, max_value=30, value=DEFAULT_WINDOW_DAYS, step=1)

st.sidebar.divider()
st.sidebar.header("Filters / Risk Controls")
trend_filter = st.sidebar.selectbox("Trend filter", ["None", "Close > 200D SMA", "50D SMA > 200D SMA"], index=0)
use_rsi = st.sidebar.checkbox("Use RSI confirmation (buy only if RSI ≤ threshold)", value=False)
rsi_period = st.sidebar.number_input("RSI period", min_value=5, max_value=100, value=DEFAULT_RSI_PERIOD, step=1)
rsi_max = st.sidebar.number_input("RSI threshold (≤)", min_value=5.0, max_value=60.0, value=DEFAULT_RSI_MAX, step=1.0)
require_new_high_reset = st.sidebar.checkbox("Require NEW rolling high after a buy before next signal", value=DEFAULT_REQUIRE_NEW_HIGH_RESET)
max_sig_per_month = st.sidebar.number_input("Max buys per month (0 = no cap)", min_value=0, max_value=30, value=DEFAULT_MAX_SIG_PER_MONTH, step=1)

st.sidebar.divider()
st.sidebar.header("Execution / Frictions")
buy_amount = st.sidebar.number_input("Buy amount per signal (USD, pre-fees)", min_value=10.0, max_value=100000.0, value=DEFAULT_BUY_AMOUNT, step=10.0)
fee_pct = st.sidebar.number_input("Fee (%) on trades", min_value=0.0, max_value=5.0, value=DEFAULT_FEE_PCT, step=0.01)
slippage_pct = st.sidebar.number_input("Slippage (%) on trades", min_value=0.0, max_value=5.0, value=DEFAULT_SLIPPAGE_PCT, step=0.01)
cooldown_days = st.sidebar.number_input("Buy cooldown (days)", min_value=0, max_value=30, value=DEFAULT_COOLDOWN_DAYS, step=1)

st.sidebar.divider()
st.sidebar.header("Take-Profit / Rebalance")
tp_use = st.sidebar.checkbox("Enable take-profit sells", value=DEFAULT_TP_USE)
tp_basis = st.sidebar.selectbox("TP basis", ["Average cost", "Last buy price"], index=0)
tp_trigger_pct = st.sidebar.number_input("TP trigger (%) above basis", min_value=1.0, max_value=500.0, value=DEFAULT_TP_TRIGGER_PCT, step=1.0)
tp_sell_pct = st.sidebar.number_input("TP sell (% of holdings)", min_value=1.0, max_value=100.0, value=DEFAULT_TP_SELL_PCT, step=1.0)
tp_cooldown_days = st.sidebar.number_input("TP cooldown (days)", min_value=0, max_value=90, value=DEFAULT_TP_COOLDOWN_DAYS, step=1)

st.sidebar.divider()
st.sidebar.header("Allocation Caps")
max_invested_usd = st.sidebar.number_input("Max total invested USD (0 = no cap)", min_value=0.0, max_value=10_000_000.0, value=DEFAULT_MAX_INVESTED_USD, step=100.0)
max_position_value_usd = st.sidebar.number_input("Max position value USD (0 = no cap)", min_value=0.0, max_value=10_000_000.0, value=DEFAULT_MAX_POSITION_VALUE_USD, step=100.0)

st.sidebar.divider()
st.sidebar.header("Starting Portfolio")
start_value_usd = st.sidebar.number_input("Starting value (USD)", min_value=0.0, value=DEFAULT_START_VALUE, step=100.0)
ref_price_usd = st.sidebar.number_input("Reference price to back into units", min_value=0.01, value=DEFAULT_REF_PRICE, step=0.01)

# ----------------------------------------------------
# Fetch & prep (single-asset panel)
# ----------------------------------------------------
primary = data_source
ohlc, used_symbol = fetch_ohlc(start_date, end_date, source=primary, asset_key=asset_key)
used_source = primary

if ohlc.empty:
    if ASSETS[asset_key]["kind"] == "crypto":
        st.warning(f"{primary} returned 0 rows; trying alternate sources…")
        for alt in ["Coinbase","Kraken","Binance (ETH/USDT)","Yahoo Finance"]:
            ohlc, used_symbol = fetch_ohlc(start_date, end_date, source=alt, asset_key=asset_key)
            if not ohlc.empty:
                used_source = alt
                break
    else:
        # Already tried Yahoo fallbacks list inside fetch_ohlc_with_symbol_fallbacks
        used_source = "Yahoo Finance"

ohlc = _finalize_ohlc(ohlc, start_date, end_date)

st.write(
    f"Using data source: {used_source} · symbol: {used_symbol or asset_key} · rows: {len(ohlc)} · "
    f"first: {ohlc.index.min() if len(ohlc) else None} · last: {ohlc.index.max() if len(ohlc) else None}"
)
if len(st.session_state.last_loader_error) and ohlc.empty:
    st.info("Last loader error: " + st.session_state.last_loader_error)

if ohlc.empty:
    st.error("No price data returned from any source for the selected dates.")
    st.stop()

# Indicators & signals
ind = compute_indicators(ohlc, window_days=int(DEFAULT_WINDOW_DAYS), rsi_period=int(DEFAULT_RSI_PERIOD), atr_period=int(DEFAULT_ATR_PERIOD))
base_signal = compute_base_signals(ind, threshold_mode=threshold_mode, fixed_pct=float(threshold_pct), atr_mult=float(atr_mult))

# Simulate (selected asset)
# For non-ETH, if you want the same initial units idea, we estimate ref from start_date close
ref_for_this = ref_price_usd if ASSETS[asset_key]["kind"] == "crypto" else price_on_or_near(pd.DataFrame({"Close": ind["Close"]}), start_date)
if pd.isna(ref_for_this) or ref_for_this <= 0:
    ref_for_this = ref_price_usd

summary, details = simulate_strategy(
    ind=ind, base_signal=base_signal, buy_amount=float(buy_amount),
    start_value_usd=float(start_value_usd), ref_price_usd=float(ref_for_this),
    fee_pct=float(fee_pct), slippage_pct=float(slippage_pct),
    cooldown_days=int(cooldown_days), trend_filter=trend_filter, use_rsi=bool(use_rsi),
    rsi_max=float(rsi_max), require_new_high_reset=bool(require_new_high_reset),
    max_signals_per_month=int(max_sig_per_month), tp_use=bool(tp_use), tp_basis=str(tp_basis),
    tp_trigger_pct=float(tp_trigger_pct), tp_sell_pct=float(tp_sell_pct), tp_cooldown_days=int(tp_cooldown_days),
    max_invested_usd=float(max_invested_usd), max_position_value_usd=float(max_position_value_usd)
)

# ----------------------------------------------------
# Output (selected asset)
# ----------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades executed", f"{summary['n_trades']}")
m2.metric("Units held (final)", f"{summary['final_units']:.6f}")
m3.metric("Ending value", format_usd(summary["terminal_value"]))
m4.metric("XIRR (annualized)" if pd.notna(summary["irr_xirr"]) else "XIRR (annualized)",
          f"{summary['irr_xirr']*100:.2f}%" if pd.notna(summary["irr_xirr"]) else "N/A")

avg_basis_now = details["ind"]["AvgCostBasis"].iloc[-1] if "AvgCostBasis" in details["ind"].columns else np.nan
a1, a2 = st.columns(2)
a1.metric("Avg cost (current)", format_usd(avg_basis_now) if pd.notna(avg_basis_now) else "—")
if ASSETS[asset_key]["kind"] == "crypto" and (used_symbol or asset_key) == "ETH-USD":
    spot_price, spot_src = fetch_spot_eth_usd()
else:
    spot_price, spot_src = fetch_spot_yahoo_close(used_symbol or asset_key)
a2.metric(f"{ASSETS[asset_key]['label'].split('(')[0].strip()} price (now)",
          f"{format_usd(spot_price) if pd.notna(spot_price) else '—'}" + (f" · {spot_src}" if spot_src and spot_src != "—" else ""))

i1, i2 = st.columns(2)
i1.metric("Total invested (incl. fees & starting)", format_usd(summary["total_invested"]))
i2.metric("P/L vs invested", f"{format_usd(summary['absolute_pnl'])} ({summary['pct_return_on_invested']:.2f}%)" if pd.notna(summary['pct_return_on_invested']) else "—")

st.divider()

st.subheader(f"{used_symbol or asset_key} Price with Buys & Sells")
fig1 = plt.figure()
plt.plot(details["ind"].index, details["ind"]["Close"].values)
trades_df = details["trades_df"]
if not trades_df.empty:
    buys = trades_df[trades_df["Type"] == "BUY"]
    sells = trades_df[trades_df["Type"] == "SELL"]
    if not buys.empty:
        bp_idx = pd.to_datetime(buys["Date"])
        plt.scatter(bp_idx, details["ind"]["Close"].loc[bp_idx].values, marker="^")
    if not sells.empty:
        sp_idx = pd.to_datetime(sells["Date"])
        plt.scatter(sp_idx, details["ind"]["Close"].loc[sp_idx].values, marker="v")
plt.title(f"{used_symbol or asset_key} (Close) — Buys (^) and Sells (v)")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
st.pyplot(fig1, use_container_width=True)

st.subheader("Average Cost Basis Over Time")
fig_cb = plt.figure()
plt.plot(details["ind"].index, details["ind"]["Close"], label=f"{used_symbol or asset_key} Close")
plt.plot(details["ind"].index, details["ind"]["AvgCostBasis"], label="Avg Cost Basis", linestyle="--")
plt.legend(); plt.title("Average Cost Basis vs Market Price")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
st.pyplot(fig_cb, use_container_width=True)

st.subheader("Portfolio Value Over Time")
fig2 = plt.figure()
plt.plot(details["portfolio_value"].index, details["portfolio_value"].values)
plt.title("Portfolio Value"); plt.xlabel("Date"); plt.ylabel("Value (USD)")
st.pyplot(fig2, use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("### Trades")
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        st.download_button("Download trades (CSV)", data=trades_df.to_csv(index=False), file_name=f"trades_{(used_symbol or asset_key)}.csv", mime="text/csv", key="dl_trades_main")
    else:
        st.info("No trades executed in the selected window.")
with right:
    st.markdown("### Cash Flows (for XIRR)")
    cf = details["cashflows"].copy()
    st.dataframe(cf, use_container_width=True, hide_index=True)
    st.download_button("Download cash flows (CSV)", data=cf.to_csv(index=False), file_name=f"cashflows_{(used_symbol or asset_key)}.csv", mime="text/csv", key="dl_cf_main")

st.divider()

# ----------------------------------------------------
# Benchmarks (Buy & Hold / Monthly DCA)
# ----------------------------------------------------
st.header("📊 Benchmarks (for comparison)")
bench_anchor = max(details["ind"].index.min().date(), DEFAULT_START_DATE)
ind_bench = details["ind"].loc[pd.Timestamp(bench_anchor):].copy()

def render_block(sim_summary, sim_details, label_prefix: str, key_prefix: str):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades executed", f"{sim_summary['n_trades']}")
    c2.metric("Units held (final)", f"{sim_summary['final_units']:.6f}")
    c3.metric("Ending value", format_usd(sim_summary["terminal_value"]))
    c4.metric("XIRR", f"{sim_summary['irr_xirr']*100:.2f}%" if pd.notna(sim_summary["irr_xirr"]) else "N/A")

    st.subheader(f"{label_prefix} {(used_symbol or asset_key)} Price with Buys & Sells")
    figp = plt.figure()
    plt.plot(sim_details["ind"].index, sim_details["ind"]["Close"].values)
    _tr = sim_details["trades_df"]
    if not _tr.empty:
        _buys = _tr[_tr["Type"] == "BUY"]; _sells = _tr[_tr["Type"] == "SELL"]
        if not _buys.empty:
            _bp_idx = pd.to_datetime(_buys["Date"]); plt.scatter(_bp_idx, sim_details["ind"]["Close"].loc[_bp_idx].values, marker="^")
        if not _sells.empty:
            _sp_idx = pd.to_datetime(_sells["Date"]); plt.scatter(_sp_idx, sim_details["ind"]["Close"].loc[_sp_idx].values, marker="v")
    plt.title(f"{label_prefix} {(used_symbol or asset_key)} (Close) — Buys (^) and Sells (v)")
    plt.xlabel("Date"); plt.ylabel("Price (USD)")
    st.pyplot(figp, use_container_width=True)

    st.subheader(f"{label_prefix} Portfolio Value Over Time")
    figv = plt.figure()
    plt.plot(sim_details["portfolio_value"].index, sim_details["portfolio_value"].values)
    plt.title(f"{label_prefix} Portfolio Value"); plt.xlabel("Date"); plt.ylabel("Value (USD)")
    st.pyplot(figv, use_container_width=True)

tabs = st.tabs([
    f"Buy & Hold — $2,500 on {bench_anchor.isoformat()}",
    f"Monthly DCA — $2,500 on {bench_anchor.isoformat()} + $250 each 27th"
])

if ind_bench.empty:
    st.info("Not enough data to compute benchmarks for the selected dates.")
else:
    with tabs[0]:
        anchor_price = price_on_or_near(ind_bench, bench_anchor)
        base_sig_bh = pd.Series(False, index=ind_bench.index)
        sum_bh, det_bh = simulate_strategy(
            ind=ind_bench, base_signal=base_sig_bh, buy_amount=0.0,
            start_value_usd=2500.0, ref_price_usd=float(anchor_price),
            fee_pct=float(fee_pct), slippage_pct=float(slippage_pct),
            cooldown_days=0, trend_filter="None", use_rsi=False, rsi_max=float(rsi_max),
            require_new_high_reset=False, max_signals_per_month=0, tp_use=False,
            tp_basis="Average cost", tp_trigger_pct=float(tp_trigger_pct), tp_sell_pct=float(tp_sell_pct),
            tp_cooldown_days=int(tp_cooldown_days), max_invested_usd=0.0, max_position_value_usd=0.0
        )
        render_block(sum_bh, det_bh, label_prefix="Buy & Hold —", key_prefix=f"bh_{(used_symbol or asset_key)}")

    with tabs[1]:
        anchor_price = price_on_or_near(ind_bench, bench_anchor)
        base_sig_dca = pd.Series(False, index=ind_bench.index)
        mask = (ind_bench.index.day == 27) & (ind_bench.index.date > bench_anchor)
        base_sig_dca.loc[mask] = True
        sum_dca, det_dca = simulate_strategy(
            ind=ind_bench, base_signal=base_sig_dca, buy_amount=250.0,
            start_value_usd=2500.0, ref_price_usd=float(anchor_price),
            fee_pct=float(fee_pct), slippage_pct=float(slippage_pct),
            cooldown_days=0, trend_filter="None", use_rsi=False, rsi_max=float(rsi_max),
            require_new_high_reset=False, max_signals_per_month=0, tp_use=False,
            tp_basis="Average cost", tp_trigger_pct=float(tp_trigger_pct), tp_sell_pct=float(tp_sell_pct),
            tp_cooldown_days=int(tp_cooldown_days), max_invested_usd=0.0, max_position_value_usd=0.0
        )
        render_block(sum_dca, det_dca, label_prefix="Monthly DCA —", key_prefix=f"dca_{(used_symbol or asset_key)}")

st.divider()

# ----------------------------------------------------
# Multi-Asset Dashboard (same parameters applied to all)
# ----------------------------------------------------
st.header("🧪 Multi-Asset Dashboard (same parameters applied to all)")
dash_tabs = st.tabs([ASSETS[k]["label"] for k in ASSETS.keys()])

def run_strategy_for_asset_key(key: str):
    df, used = fetch_ohlc(start_date, end_date, data_source, key)
    if df.empty:
        return None, None, None, None
    df = _finalize_ohlc(df, start_date, end_date)
    if df.empty:
        return None, None, None, None
    ii = compute_indicators(df, window_days=int(DEFAULT_WINDOW_DAYS), rsi_period=int(DEFAULT_RSI_PERIOD), atr_period=int(DEFAULT_ATR_PERIOD))
    sig = compute_base_signals(ii, threshold_mode=threshold_mode, fixed_pct=float(threshold_pct), atr_mult=float(atr_mult))
    ref_guess = price_on_or_near(pd.DataFrame({"Close": ii["Close"]}), start_date)
    start_ref = float(ref_price_usd if key == "ETH-USD" else (ref_guess if not pd.isna(ref_guess) else ref_price_usd))
    summ, det = simulate_strategy(
        ind=ii, base_signal=sig, buy_amount=float(buy_amount),
        start_value_usd=float(start_value_usd), ref_price_usd=float(start_ref),
        fee_pct=float(fee_pct), slippage_pct=float(slippage_pct),
        cooldown_days=int(cooldown_days), trend_filter=trend_filter, use_rsi=bool(use_rsi),
        rsi_max=float(rsi_max), require_new_high_reset=bool(require_new_high_reset),
        max_signals_per_month=int(max_sig_per_month), tp_use=bool(tp_use), tp_basis=str(tp_basis),
        tp_trigger_pct=float(tp_trigger_pct), tp_sell_pct=float(tp_sell_pct), tp_cooldown_days=int(tp_cooldown_days),
        max_invested_usd=float(max_invested_usd), max_position_value_usd=float(max_position_value_usd)
    )
    return used, ii, summ, det

compare_rows = []
per_asset_results = {}
for i, key in enumerate(ASSETS.keys()):
    used_sym, ii, summ, det = run_strategy_for_asset_key(key)
    with dash_tabs[i]:
        label = ASSETS[key]["label"]
        if ii is None:
            st.error(f"No data for {label} in the selected range (tried: {', '.join(ASSETS[key].get('symbols', [key]))}).")
        else:
            per_asset_results[key] = (used_sym, ii, summ, det)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", f"{summ['n_trades']}")
            c2.metric("Units (final)", f"{summ['final_units']:.6f}")
            c3.metric("End value", format_usd(summ["terminal_value"]))
            c4.metric("XIRR", f"{summ['irr_xirr']*100:.2f}%" if pd.notna(summ["irr_xirr"]) else "N/A")
            st.caption(f"Symbol used: **{used_sym}**")

            figd = plt.figure()
            plt.plot(det["ind"].index, det["ind"]["Close"].values)
            trd = det["trades_df"]
            if not trd.empty:
                bs = trd[trd["Type"]=="BUY"]; ss = trd[trd["Type"]=="SELL"]
                if not bs.empty:
                    _b = pd.to_datetime(bs["Date"]); plt.scatter(_b, det["ind"]["Close"].loc[_b].values, marker="^")
                if not ss.empty:
                    _s = pd.to_datetime(ss["Date"]); plt.scatter(_s, det["ind"]["Close"].loc[_s].values, marker="v")
            plt.title(f"{used_sym} — Close with Buys (^) / Sells (v)")
            plt.xlabel("Date"); plt.ylabel("Price (USD)")
            st.pyplot(figd, use_container_width=True)

            compare_rows.append({
                "AssetKey": key,
                "UsedSymbol": used_sym,
                "Trades": int(summ["n_trades"]),
                "Units Final": float(summ["final_units"]),
                "Terminal Value (USD)": float(summ["terminal_value"]),
                "Total Invested (USD)": float(summ["total_invested"]),
                "Abs P&L (USD)": float(summ["absolute_pnl"]),
                "P&L % on Invested": float(summ["pct_return_on_invested"]) if pd.notna(summ["pct_return_on_invested"]) else np.nan,
                "XIRR": float(summ["irr_xirr"]) if pd.notna(summ["irr_xirr"]) else np.nan,
            })

if compare_rows:
    cmp_df = pd.DataFrame(compare_rows).set_index("AssetKey")
    st.subheader("Multi-Asset Comparison (same parameters)")
    st.dataframe(cmp_df, use_container_width=True)
    st.download_button("Download comparison (CSV)", data=cmp_df.to_csv(), file_name="multi_asset_comparison.csv", mime="text/csv", key="dl_multi_cmp")

st.divider()

# ----------------------------------------------------
# Correlation & Beta (1y / 3y / 5y) — pairwise daily returns
# ----------------------------------------------------
st.header("📈 Correlation & Beta (1y / 3y / 5y)")

def load_close_series_for_all(end_dt: date, years: int):
    start_dt = end_dt - timedelta(days=365*years + 14)  # buffer
    frames = {}
    symbols_used = {}
    for key in ASSETS.keys():
        df, used = fetch_ohlc(start_dt, end_dt, data_source, key)
        if df.empty:
            continue
        df = _finalize_ohlc(df, start_dt, end_dt)
        if not df.empty:
            frames[key] = df["Close"].rename(key)  # keep asset key as column, not symbol
            symbols_used[key] = used
    if not frames:
        return pd.DataFrame(), {}
    allc = pd.concat(frames.values(), axis=1).dropna(how="any")
    return allc, symbols_used

def corr_beta_tables(end_dt: date, years: int):
    closes, sym_used = load_close_series_for_all(end_dt, years)
    if closes.empty or len(closes) < 30:
        return None, None, sym_used
    rets = closes.pct_change().dropna(how="any")
    corr = rets.corr()
    cov = rets.cov()
    var = np.diag(cov.values)
    beta = pd.DataFrame(index=cov.index, columns=cov.columns, dtype=float)
    for i in cov.index:
        for j_idx, j in enumerate(cov.columns):
            denom = var[j_idx]
            beta.loc[i, j] = (cov.loc[i, j] / denom) if denom != 0 else np.nan
    return corr, beta, sym_used

for years in [1, 3, 5]:
    corr, beta, used_map = corr_beta_tables(end_date, years)
    st.subheader(f"Time Horizon: {years} year{'s' if years>1 else ''}")
    if corr is None:
        st.info("Not enough overlapping data to compute metrics for this horizon.")
    else:
        st.caption("Symbols used: " + ", ".join([f"{k}→{v}" for k,v in used_map.items() if v]))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Correlation (daily returns)**")
            st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
            st.download_button(f"Download correlation {years}y (CSV)", data=corr.to_csv(), file_name=f"corr_{years}y.csv", mime="text/csv", key=f"dl_corr_{years}")
        with c2:
            st.markdown("**Beta matrix βᵢ⟂ⱼ (daily returns)**")
            st.caption("β of row asset *i* with respect to column asset *j* (Cov(i,j)/Var(j)).")
            st.dataframe(beta.style.format("{:.2f}"), use_container_width=True)
            st.download_button(f"Download beta {years}y (CSV)", data=beta.to_csv(), file_name=f"beta_{years}y.csv", mime="text/csv", key=f"dl_beta_{years}")

st.divider()

# ----------------------------------------------------
# Methodology
# ----------------------------------------------------
st.markdown("""
**Methodology & Controls**
- Dip buy: first cross where drawdown vs rolling window high ≥ threshold (fixed % or ATR×mult).
- Filters: Trend (SMA), RSI, new-high reset, cooldown, monthly cap.
- Execution: Slippage & fees on buys/sells. Average-cost basis tracked daily.
- Take-profit: when price ≥ basis × (1 + TP%), sell % with cooldown.
- Caps: skip if total invested or position value would exceed caps.
- Performance: XIRR from cash flows.
- Data: ETH uses Coinbase/Kraken/Binance/Yahoo with fallback; GLD/^SOX/ARTY use Yahoo with **symbol fallbacks**.
- Benchmarks: Buy & Hold and Monthly DCA from 2020-09-27 anchor where possible.
- Correlation/Beta: daily pct-change returns; βᵢ⟂ⱼ = Cov(i,j)/Var(j) (row vs column).
""")
