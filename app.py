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
# Defaults (your specs)
# --------------------------
DEFAULT_START_DATE = date(2020, 9, 27)
DEFAULT_FIXED_END_DATE = date(2025, 8, 27)

# Core dip heuristic
DEFAULT_THRESHOLD_PCT = 5.0
DEFAULT_WINDOW_DAYS = 7
DEFAULT_BUY_AMOUNT = 150.0

# Starting portfolio
DEFAULT_START_VALUE = 2500.00
DEFAULT_REF_PRICE   = 354.31

# Trading frictions
DEFAULT_FEE_PCT = 0.10
DEFAULT_SLIPPAGE_PCT = 0.05
DEFAULT_COOLDOWN_DAYS = 0

# Improvement options
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULT   = 2.0
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_MAX    = 45.0
DEFAULT_REQUIRE_NEW_HIGH_RESET = False
DEFAULT_MAX_SIG_PER_MONTH = 0

# Take-profit / rebalance
DEFAULT_TP_USE = False
DEFAULT_TP_BASIS = "Average cost"
DEFAULT_TP_TRIGGER_PCT = 20.0
DEFAULT_TP_SELL_PCT = 10.0
DEFAULT_TP_COOLDOWN_DAYS = 7

# Allocation caps
DEFAULT_MAX_INVESTED_USD = 0.0
DEFAULT_MAX_POSITION_VALUE_USD = 0.0

# Debug bucket
if "last_loader_error" not in st.session_state:
    st.session_state.last_loader_error = ""

# --------------------------
# Helpers
# --------------------------
def _std_headers():
    return {
        "User-Agent": "Mozilla/5.0 (compatible; ETHDipApp/1.0; +https://streamlit.io)",
        "Accept": "application/json",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

def _finalize_ohlc(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Standardize index, sort, slice, and DROP DUPLICATES."""
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    df = df[~df.index.duplicated(keep="last")]
    keep = [c for c in ["High", "Low", "Close"] if c in df.columns]
    df = df[keep].dropna(how="any")
    return df

# --------------------------
# Data loaders (Yahoo, CoinGecko, Binance, Coinbase, Kraken)
# --------------------------
@st.cache_data(show_spinner=False)
def fetch_ohlc_yahoo(start: date, end: date) -> pd.DataFrame:
    import time
    try:
        df = None
        for _ in range(3):
            try:
                df = yf.download(
                    "ETH-USD",
                    start=start,
                    end=end + timedelta(days=1),  # yfinance end is exclusive
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                )
                if df is not None and not df.empty:
                    break
            except Exception as e:
                st.session_state.last_loader_error = f"Yahoo error: {e!s}"
            time.sleep(1.0)

        if df is None or len(df) == 0:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([c for c in col if c]) for c in df.columns]

        def pick_col(name):
            if name in df.columns:
                return df[name]
            cands = [c for c in df.columns if c.split("_")[-1] == name or c.split(" ")[-1] == name]
            return df[cands[0]] if cands else pd.Series(index=df.index, dtype="float64")

        out = pd.DataFrame({
            "High": pd.to_numeric(pick_col("High"), errors="coerce"),
            "Low":  pd.to_numeric(pick_col("Low"),  errors="coerce"),
            "Close":pd.to_numeric(pick_col("Close"),errors="coerce"),
        })
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Yahoo fatal: {e!s}"
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_coingecko(start: date, end: date) -> pd.DataFrame:
    import requests
    try:
        url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
        params = {"vs_currency": "usd", "days": "max"}
        r = requests.get(url, params=params, timeout=30, headers=_std_headers())
        r.raise_for_status()
        data = r.json()
        if "prices" not in data or len(data["prices"]) == 0:
            st.session_state.last_loader_error = "CoinGecko returned no 'prices'."
            return pd.DataFrame()

        prices = pd.DataFrame(data["prices"], columns=["ts_ms", "price"])
        prices["ts"] = pd.to_datetime(prices["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
        prices = prices.set_index("ts").sort_index()

        start_ts = pd.to_datetime(start) - pd.Timedelta(days=1)
        end_ts   = pd.to_datetime(end)   + pd.Timedelta(days=1)
        prices = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)]

        daily = prices["price"].resample("D").agg(["max", "min", "last"]).dropna(how="any")
        daily.rename(columns={"max": "High", "min": "Low", "last": "Close"}, inplace=True)
        out = daily[["High", "Low", "Close"]].copy()
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"CoinGecko error: {e!s}"
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_binance(start: date, end: date) -> pd.DataFrame:
    import requests, time
    try:
        start_ts = int(pd.Timestamp(start).timestamp() * 1000)
        end_ts   = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000)
        frames, cur = [], start_ts
        while cur < end_ts:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol":"ETHUSDT","interval":"1d","startTime":cur,"endTime":end_ts,"limit":1000}
            r = requests.get(url, params=params, timeout=30, headers=_std_headers())
            if r.status_code == 451:
                st.session_state.last_loader_error = f"Binance HTTP {r.status_code}"
                return pd.DataFrame()
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
            last_open = int(klines[-1][0])
            cur = last_open + 24*60*60*1000
            time.sleep(0.25)
        if not frames:
            st.session_state.last_loader_error = "Binance returned no klines."
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Binance error: {e!s}"
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
            url = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
            params = {"granularity":86400, "start":cur.isoformat(), "end":chunk_end.isoformat()}
            r = requests.get(url, params=params, timeout=30, headers=_std_headers())
            if r.status_code == 429:
                time.sleep(1.0)
                r = requests.get(url, params=params, timeout=30, headers=_std_headers())
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
            st.session_state.last_loader_error = "Coinbase returned no candles."
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Coinbase error: {e!s}"
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlc_kraken(start: date, end: date) -> pd.DataFrame:
    import requests, time
    try:
        frames = []
        cur = int(pd.Timestamp(start).timestamp())
        end_sec = int(pd.Timestamp(end + timedelta(days=1)).timestamp())
        while cur < end_sec:
            url = "https://api.kraken.com/0/public/OHLC"
            params = {"pair":"ETHUSD","interval":1440,"since":cur}
            r = requests.get(url, params=params, timeout=30, headers=_std_headers())
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                st.session_state.last_loader_error = f"Kraken error: {data['error']}"
                break
            result = data.get("result", {})
            key = next((k for k in result.keys() if k not in ("last")), None)
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
            last_ts = int(rows[-1][0])
            cur = last_ts + 24*60*60
            time.sleep(0.25)
        if not frames:
            st.session_state.last_loader_error = "Kraken returned no OHLC."
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return _finalize_ohlc(out, start, end)
    except Exception as e:
        st.session_state.last_loader_error = f"Kraken error: {e!s}"
        return pd.DataFrame()

def fetch_ohlc(start: date, end: date, source: str) -> pd.DataFrame:
    order = {
        "Yahoo Finance":         [fetch_ohlc_yahoo,   fetch_ohlc_coinbase, fetch_ohlc_kraken,  fetch_ohlc_binance,  fetch_ohlc_coingecko],
        "CoinGecko":             [fetch_ohlc_coingecko, fetch_ohlc_coinbase, fetch_ohlc_kraken, fetch_ohlc_binance, fetch_ohlc_yahoo],
        "Binance (ETH/USDT)":    [fetch_ohlc_binance, fetch_ohlc_coinbase, fetch_ohlc_kraken,  fetch_ohlc_coingecko, fetch_ohlc_yahoo],
        "Coinbase":              [fetch_ohlc_coinbase, fetch_ohlc_kraken,  fetch_ohlc_binance,  fetch_ohlc_coingecko, fetch_ohlc_yahoo],
        "Kraken":                [fetch_ohlc_kraken,  fetch_ohlc_coinbase, fetch_ohlc_binance,  fetch_ohlc_coingecko, fetch_ohlc_yahoo],
    }.get(source, [fetch_ohlc_coinbase, fetch_ohlc_kraken, fetch_ohlc_yahoo, fetch_ohlc_binance, fetch_ohlc_coingecko])

    for fn in order:
        df = fn(start, end)
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()

# --------------------------
# Indicators & signals
# --------------------------
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
    """Stable XIRR via bisection. Merges same-day cashflows, requires both signs."""
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


# --------------------------
# Simulation (index-safe iteration) + Avg Cost Basis tracking
# --------------------------
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

    # Align signal & coerce to bool array (position-based)
    if not isinstance(base_signal, pd.Series):
        base_signal = pd.Series(base_signal, index=idx)
    signal = base_signal.reindex(idx, fill_value=False).astype(bool).to_numpy()

    close    = ind["Close"].to_numpy()
    roll_max = ind["RollingMax"].to_numpy()
    sma50    = ind["SMA50"].to_numpy()
    sma200   = ind["SMA200"].to_numpy()
    rsi      = ind["RSI"].to_numpy()
    drawdown = ind["DrawdownPct"].to_numpy()

    # state
    current_units = start_value_usd / ref_price_usd
    total_cost_excl_fees = start_value_usd
    last_buy_price = ref_price_usd
    last_buy_date = None
    last_tp_date  = None
    last_signal_rollmax_at_buy = None
    month_counts = {}

    # cashflows
    cashflows_dates = [idx[0].date()]
    cashflows_amounts = [-float(start_value_usd)]
    trades = []

    # time series we track
    units_series = pd.Series(index=idx, dtype=float)
    cost_basis_series = pd.Series(index=idx, dtype=float)

    def avg_cost_per_unit():
        return total_cost_excl_fees / current_units if current_units > 0 else np.nan

    for i, dt in enumerate(idx):
        # ----- take-profit first -----
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

        # ----- buy logic -----
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

        # trend filter
        if execute_buy and trend_filter == "Close > 200D SMA":
            execute_buy = bool(close[i] > sma200[i])
        elif execute_buy and trend_filter == "50D SMA > 200D SMA":
            execute_buy = bool(sma50[i] > sma200[i])

        # rsi filter
        if execute_buy and use_rsi:
            rsi_val = rsi[i]
            if pd.isna(rsi_val) or not (rsi_val <= float(rsi_max)):
                execute_buy = False

        # allocation caps
        if execute_buy:
            total_invested_so_far = -sum(a for a in cashflows_amounts if a < 0)
            prospective_cash_out = float(buy_amount) * (1.0 + float(fee_pct) / 100.0)
            if (max_invested_usd > 0) and (total_invested_so_far + prospective_cash_out > float(max_invested_usd)):
                execute_buy = False
            position_value_now = current_units * float(close[i])
            if (max_position_value_usd > 0) and (position_value_now > float(max_position_value_usd)):
                execute_buy = False

        # execute BUY
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

        # record daily state
        units_series.iloc[i] = current_units
        cost_basis_series.iloc[i] = (total_cost_excl_fees / current_units) if current_units > 0 else np.nan

    # attach series
    ind = ind.copy()
    ind["Units"] = units_series
    ind["AvgCostBasis"] = cost_basis_series

    # terminal metrics
    portfolio_value = ind["Units"] * ind["Close"]
    terminal_value = float(portfolio_value.iloc[-1])
    cashflows_dates.append(idx[-1].date())
    cashflows_amounts.append(terminal_value)

    irr = robust_xirr(pd.to_datetime(cashflows_dates), cashflows_amounts)

    total_invested = float(-sum(a for a in cashflows_amounts if a < 0))
    summary = {
        "initial_eth": float((DEFAULT_START_VALUE if start_value_usd is None else start_value_usd) / ref_price_usd),
        "final_eth": float(ind["Units"].iloc[-1]),
        "terminal_value": float(terminal_value),
        "total_invested": total_invested,
        "n_trades": int(len(trades)),
        "irr_xirr": irr,
        "absolute_pnl": float(terminal_value - total_invested),
        "pct_return_on_invested": float((terminal_value / total_invested - 1.0) * 100.0),
    }
    details = {
        "portfolio_value": portfolio_value,
        "trades_df": pd.DataFrame(trades),
        "cashflows": pd.DataFrame({"Date": cashflows_dates, "Amount": cashflows_amounts}),
        "ind": ind,
    }
    return summary, details

def format_usd(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

# --------------------------
# UI
# --------------------------
st.title("📉 ETH Buy-on-Dips — with Risk Controls")
st.caption("Educational tool. Data: Yahoo, CoinGecko, Binance, Coinbase, Kraken (auto-fallback). Not financial advice.")

with st.sidebar:
    st.header("Backtest Period")
    start_date = st.date_input("Start date", value=DEFAULT_START_DATE)
    end_mode = st.radio("End date mode", ["Rolling (use today's date)", "Fixed date"], index=0)
    end_date = date.today() if end_mode.startswith("Rolling") else st.date_input("Fixed end date", value=DEFAULT_FIXED_END_DATE, min_value=start_date)
    if end_mode.startswith("Rolling"):
        st.info(f"Using today's date: {end_date.isoformat()}")

    data_source = st.selectbox(
        "Data source",
        ["Coinbase", "Kraken", "Yahoo Finance", "Binance (ETH/USDT)", "CoinGecko"],
        index=0,
        help="Pick a source; the app will fall back through others automatically."
    )
    if st.button("🔄 Clear data cache"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.header("Signal Logic")
    threshold_mode = st.selectbox("Threshold mode", ["Fixed % (classic 5%)", "ATR multiple (vol-adaptive)"], index=0)
    if threshold_mode.startswith("ATR"):
        atr_mult = st.number_input("ATR multiple", min_value=0.5, max_value=5.0, value=DEFAULT_ATR_MULT, step=0.1)
        threshold_pct = DEFAULT_THRESHOLD_PCT
    else:
        atr_mult = DEFAULT_ATR_MULT
        threshold_pct = st.number_input("Dip threshold (%)", min_value=1.0, max_value=50.0, value=DEFAULT_THRESHOLD_PCT, step=0.5)
    window_days = st.number_input("Week window (days)", min_value=2, max_value=30, value=DEFAULT_WINDOW_DAYS, step=1)

    st.divider()
    st.header("Filters / Risk Controls")
    trend_filter = st.selectbox("Trend filter", ["None", "Close > 200D SMA", "50D SMA > 200D SMA"], index=0)
    use_rsi = st.checkbox("Use RSI confirmation (buy only if RSI ≤ threshold)", value=False)
    rsi_period = st.number_input("RSI period", min_value=5, max_value=100, value=DEFAULT_RSI_PERIOD, step=1)
    rsi_max = st.number_input("RSI threshold (≤)", min_value=5.0, max_value=60.0, value=DEFAULT_RSI_MAX, step=1.0)
    require_new_high_reset = st.checkbox("Require NEW rolling high after a buy before next signal", value=DEFAULT_REQUIRE_NEW_HIGH_RESET)
    max_sig_per_month = st.number_input("Max buys per month (0 = no cap)", min_value=0, max_value=30, value=DEFAULT_MAX_SIG_PER_MONTH, step=1)

    st.divider()
    st.header("Execution / Frictions")
    buy_amount = st.number_input("Buy amount per signal (USD, pre-fees)", min_value=10.0, max_value=100000.0, value=DEFAULT_BUY_AMOUNT, step=10.0)
    fee_pct = st.number_input("Fee (%) on trades", min_value=0.0, max_value=5.0, value=DEFAULT_FEE_PCT, step=0.01)
    slippage_pct = st.number_input("Slippage (%) on trades", min_value=0.0, max_value=5.0, value=DEFAULT_SLIPPAGE_PCT, step=0.01)
    cooldown_days = st.number_input("Buy cooldown (days)", min_value=0, max_value=30, value=DEFAULT_COOLDOWN_DAYS, step=1)

    st.divider()
    st.header("Take-Profit / Rebalance")
    tp_use = st.checkbox("Enable take-profit sells", value=DEFAULT_TP_USE)
    tp_basis = st.selectbox("TP basis", ["Average cost", "Last buy price"], index=0)
    tp_trigger_pct = st.number_input("TP trigger (%) above basis", min_value=1.0, max_value=500.0, value=DEFAULT_TP_TRIGGER_PCT, step=1.0)
    tp_sell_pct = st.number_input("TP sell (% of holdings)", min_value=1.0, max_value=100.0, value=DEFAULT_TP_SELL_PCT, step=1.0)
    tp_cooldown_days = st.number_input("TP cooldown (days)", min_value=0, max_value=90, value=DEFAULT_TP_COOLDOWN_DAYS, step=1)

    st.divider()
    st.header("Allocation Caps")
    max_invested_usd = st.number_input("Max total invested USD (0 = no cap)", min_value=0.0, max_value=10_000_000.0, value=DEFAULT_MAX_INVESTED_USD, step=100.0)
    max_position_value_usd = st.number_input("Max position value USD (0 = no cap)", min_value=0.0, max_value=10_000_000.0, value=DEFAULT_MAX_POSITION_VALUE_USD, step=100.0)

    st.divider()
    st.header("Starting Portfolio")
    start_value_usd = st.number_input("Starting value (USD)", min_value=0.0, value=DEFAULT_START_VALUE, step=100.0)
    ref_price_usd = st.number_input("Reference price to back into ETH units", min_value=0.01, value=DEFAULT_REF_PRICE, step=0.01)

# Fetch & prep with robust fallback
primary = data_source
ohlc = fetch_ohlc(start_date, end_date, source=primary)
used_source = primary
if ohlc.empty:
    st.warning(f"{primary} returned 0 rows; trying other sources…")
    for alt in ["Coinbase","Kraken","Yahoo Finance","Binance (ETH/USDT)","CoinGecko"]:
        if alt == primary:
            continue
        ohlc = fetch_ohlc(start_date, end_date, source=alt)
        if not ohlc.empty:
            used_source = alt
            break

# Final defensive dedupe (just in case)
ohlc = _finalize_ohlc(ohlc, start_date, end_date)

st.write(
    f"Using data source: {used_source} · rows: {len(ohlc)} · "
    f"first: {ohlc.index.min() if len(ohlc) else None} · last: {ohlc.index.max() if len(ohlc) else None}"
)
if len(st.session_state.last_loader_error) and ohlc.empty:
    st.info("Last loader error: " + st.session_state.last_loader_error)

if ohlc.empty:
    st.error("No price data returned from any source for the selected dates.")
    st.stop()

# Indicators & signals (use sidebar values)
ind = compute_indicators(
    ohlc,
    window_days=int(st.session_state.get("window_days", 0) or
                    locals().get("window_days", DEFAULT_WINDOW_DAYS)),
    rsi_period=int(rsi_period),
    atr_period=int(DEFAULT_ATR_PERIOD),
)

base_signal = compute_base_signals(
    ind,
    threshold_mode=threshold_mode,
    fixed_pct=float(threshold_pct),
    atr_mult=float(atr_mult)
)

# Simulate
summary, details = simulate_strategy(
    ind=ind,
    base_signal=base_signal,
    buy_amount=float(buy_amount),
    start_value_usd=float(start_value_usd),
    ref_price_usd=float(ref_price_usd),
    fee_pct=float(fee_pct),
    slippage_pct=float(slippage_pct),
    cooldown_days=int(cooldown_days),
    trend_filter=trend_filter,
    use_rsi=bool(use_rsi),
    rsi_max=float(rsi_max),
    require_new_high_reset=bool(require_new_high_reset),
    max_signals_per_month=int(max_sig_per_month),
    tp_use=bool(tp_use),
    tp_basis=str(tp_basis),
    tp_trigger_pct=float(tp_trigger_pct),
    tp_sell_pct=float(tp_sell_pct),
    tp_cooldown_days=int(tp_cooldown_days),
    max_invested_usd=float(max_invested_usd),
    max_position_value_usd=float(max_position_value_usd)
)

# --------------------------
# Output
# --------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades executed", f"{summary['n_trades']}")
m2.metric("ETH held (final)", f"{summary['final_eth']:.6f}")
m3.metric("Ending value", format_usd(summary["terminal_value"]))
m4.metric("XIRR (annualized)" if pd.notna(summary["irr_xirr"]) else "XIRR (annualized)",
          f"{summary['irr_xirr']*100:.2f}%" if pd.notna(summary["irr_xirr"]) else "N/A")

# New: show current Avg Cost Basis metric
avg_basis_now = details["ind"]["AvgCostBasis"].iloc[-1] if "AvgCostBasis" in details["ind"].columns else np.nan
st.metric("Avg cost (current)", format_usd(avg_basis_now) if pd.notna(avg_basis_now) else "—")

m5, m6 = st.columns(2)
m5.metric("Total invested (incl. fees & starting)", format_usd(summary["total_invested"]))
m6.metric("P/L vs invested", f"{format_usd(summary['absolute_pnl'])} ({summary['pct_return_on_invested']:.2f}%)")

st.divider()

st.subheader("ETH Price with Buys & Sells")
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
plt.title("ETH-USD (Close) — Buys (^) and Sells (v)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
st.pyplot(fig1, use_container_width=True)

# New: Average Cost Basis chart
st.subheader("Average Cost Basis Over Time")
fig_cb = plt.figure()
plt.plot(details["ind"].index, details["ind"]["Close"], label="ETH Close")
plt.plot(details["ind"].index, details["ind"]["AvgCostBasis"], label="Avg Cost Basis", linestyle="--")
plt.legend()
plt.title("Average Cost Basis vs Market Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
st.pyplot(fig_cb, use_container_width=True)

st.subheader("Portfolio Value Over Time")
fig2 = plt.figure()
plt.plot(details["portfolio_value"].index, details["portfolio_value"].values)
plt.title("Portfolio Value")
plt.xlabel("Date")
plt.ylabel("Value (USD)")
st.pyplot(fig2, use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("### Trades")
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        st.download_button("Download trades (CSV)", data=trades_df.to_csv(index=False), file_name="trades.csv", mime="text/csv")
    else:
        st.info("No trades executed in the selected window.")

with right:
    st.markdown("### Cash Flows (for XIRR)")
    cf = details["cashflows"].copy()
    st.dataframe(cf, use_container_width=True, hide_index=True)
    st.download_button("Download cash flows (CSV)", data=cf.to_csv(index=False), file_name="cashflows.csv", mime="text/csv")

st.divider()
st.markdown(
    """
**Methodology & Controls**
- Dip buy: first cross where drawdown vs rolling 7-day high ≥ threshold (fixed 5% or ATR×mult).
- Filters: Trend (SMA), RSI, new-high reset, cooldown, monthly cap.
- Execution: Slippage & fees on buys/sells. Average-cost basis.
- Take-profit: when price ≥ basis × (1 + TP%), sell % with cooldown.
- Caps: skip if total invested or position value would exceed caps.
- Performance: XIRR from cash flows.
- Data sources: Coinbase, Kraken, Yahoo, Binance, CoinGecko — with automatic fallback.
"""
)
