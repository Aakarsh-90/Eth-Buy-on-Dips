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
DEFAULT_START_DATE = date(2020, 9, 27)  # your first buy date
DEFAULT_FIXED_END_DATE = date(2025, 8, 27)  # can switch to rolling "today" in the sidebar

# Core dip heuristic
DEFAULT_THRESHOLD_PCT = 5.0      # classic 5% vs recent high
DEFAULT_WINDOW_DAYS = 7          # "in the week"
DEFAULT_BUY_AMOUNT = 150.0       # $150 per signal (pre-fees)

# Starting portfolio
DEFAULT_START_VALUE = 2500.00    # your first purchase (USD)
DEFAULT_REF_PRICE = 354.31       # your first buy price per ETH

# Trading frictions
DEFAULT_FEE_PCT = 0.10           # % fee on both buys & sells (applied to cash amount)
DEFAULT_SLIPPAGE_PCT = 0.05      # % slippage on both buys & sells (buys: +, sells: -)
DEFAULT_COOLDOWN_DAYS = 0        # min days between buys

# Improvement options
DEFAULT_THRESHOLD_MODE = "Fixed % (classic 5%)"  # or "ATR multiple (vol-adaptive)"
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULT = 2.0            # drawdown threshold = ATR% * MULT
DEFAULT_TREND_FILTER = "None"     # "None", "Close > 200D SMA", "50D SMA > 200D SMA"
DEFAULT_RSI_USE = False
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_MAX = 45.0            # only buy if RSI <= this
DEFAULT_REQUIRE_NEW_HIGH_RESET = False
DEFAULT_MAX_SIG_PER_MONTH = 0     # 0 = unlimited

# NEW: Take-profit / rebalance
DEFAULT_TP_USE = False
DEFAULT_TP_BASIS = "Average cost"     # "Average cost" or "Last buy price"
DEFAULT_TP_TRIGGER_PCT = 20.0         # trigger when price >= basis * (1 + pct)
DEFAULT_TP_SELL_PCT = 10.0            # sell % of current units on trigger
DEFAULT_TP_COOLDOWN_DAYS = 7          # min days between TP sells

# NEW: Allocation caps
DEFAULT_MAX_INVESTED_USD = 0.0        # 0 = no cap; cap applies to total invested (incl. starting & fees)
DEFAULT_MAX_POSITION_VALUE_USD = 0.0  # 0 = no cap; cap on holdings_value = units * price

# --------------------------
# Data & indicators
# --------------------------
@st.cache_data(show_spinner=False)
def fetch_ohlc(start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLC for ETH-USD from Yahoo Finance."""
    data = yf.download(
        "ETH-USD",
        start=start,
        end=end + timedelta(days=1),  # yfinance end is exclusive
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        return pd.DataFrame()
    df = data[["High", "Low", "Close"]].dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[(df.index.date >= start) & (df.index.date <= end)]
    return df

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def compute_indicators(ohlc: pd.DataFrame, rsi_period=14, atr_period=14):
    close = ohlc["Close"]
    high, low = ohlc["High"], ohlc["Low"]

    roll_max = close.rolling(window=DEFAULT_WINDOW_DAYS, min_periods=1).max()
    drawdown = (roll_max - close) / roll_max  # fraction

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi = rsi_wilder(close, rsi_period)
    atr = atr_wilder(high, low, close, atr_period)
    atr_pct = (atr / close) * 100.0

    out = pd.DataFrame({
        "Close": close,
        "High": high,
        "Low": low,
        "RollingMax": roll_max,
        "DrawdownPct": drawdown * 100.0,
        "SMA50": sma50,
        "SMA200": sma200,
        "RSI": rsi,
        "ATR": atr,
        "ATR_Pct": atr_pct
    })
    return out

def compute_base_signals(ind: pd.DataFrame, threshold_mode: str, fixed_pct: float, atr_mult: float) -> pd.Series:
    """Return base 'crossed' signal (before filters) when drawdown crosses above threshold."""
    if threshold_mode.startswith("ATR"):
        thr_series = ind["ATR_Pct"] * atr_mult
    else:
        thr_series = pd.Series(fixed_pct, index=ind.index)

    cond = ind["DrawdownPct"] >= thr_series
    crossed = cond & (~cond.shift(1).fillna(False))
    return crossed

# --------------------------
# Simulation (buys, sells, caps)
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
    # Take-profit
    tp_use: bool = False,
    tp_basis: str = "Average cost",
    tp_trigger_pct: float = 20.0,
    tp_sell_pct: float = 10.0,
    tp_cooldown_days: int = 7,
    # Allocation caps
    max_invested_usd: float = 0.0,
    max_position_value_usd: float = 0.0
):
    """
    - Starting ETH units = start_value_usd / ref_price_usd; starting cost basis = start_value_usd.
    - Buys use base_signal + filters; sells via take-profit rule.
    - Fees/slippage on both sides:
        * Buy executed_price = Close * (1 + slippage%), cash out = buy_amount + fee%*buy_amount.
        * Sell executed_price = Close * (1 - slippage%), proceeds net of fee% on gross proceeds.
    - Average-cost accounting for basis.
    - Caps:
        * max_invested_usd caps total negative cashflows (including starting value & fees).
        * max_position_value_usd caps units*price — if exceeded, skip new buys.
    """
    idx = ind.index
    close = ind["Close"]
    roll_max = ind["RollingMax"]

    # --- State ---
    current_units = start_value_usd / ref_price_usd
    total_cost_excl_fees = start_value_usd  # average cost accounting
    last_buy_price = ref_price_usd          # track last executed buy price
    last_buy_date = None
    last_tp_date = None
    last_signal_rollmax_at_buy = None
    month_counts = {}  # (year, month) -> #buys

    # Cash flows (XIRR): negative=outflow, positive=inflow
    cashflows_dates = [idx[0].date()]
    cashflows_amounts = [-float(start_value_usd)]

    # Logs
    trades = []  # unified list of buys & sells

    def avg_cost_per_unit():
        return total_cost_excl_fees / current_units if current_units > 0 else np.nan

    for dt in idx:
        # --- Take-profit first (optional) ---
        if tp_use and current_units > 0:
            # check TP cooldown
            can_tp = (last_tp_date is None) or ((dt.date() - last_tp_date).days >= int(tp_cooldown_days))

            if can_tp:
                basis_price = avg_cost_per_unit() if tp_basis == "Average cost" else last_buy_price
                if not pd.isna(basis_price):
                    trigger_price = basis_price * (1.0 + float(tp_trigger_pct) / 100.0)
                    if close.loc[dt] >= trigger_price:
                        sell_units = current_units * (float(tp_sell_pct) / 100.0)
                        if sell_units > 0:
                            exec_price = float(close.loc[dt]) * (1.0 - float(slippage_pct) / 100.0)
                            gross = sell_units * exec_price
                            fee = gross * (float(fee_pct) / 100.0)
                            net = gross - fee

                            # update holdings & basis (average-cost method)
                            # remove cost proportional to units sold
                            nonlocal_total_cost = total_cost_excl_fees  # for mypy clarity
                            cost_removed = avg_cost_per_unit() * sell_units if not pd.isna(avg_cost_per_unit()) else 0.0
                            total_cost_excl_fees -= cost_removed
                            current_units -= sell_units

                            # record cashflow (positive)
                            cashflows_dates.append(dt.date())
                            cashflows_amounts.append(net)

                            trades.append({
                                "Type": "SELL",
                                "Date": dt.date(),
                                "Price_Close": float(close.loc[dt]),
                                "Executed_Price": exec_price,
                                "Units": -sell_units,
                                "Gross_Proceeds": gross,
                                "Fee_%": float(fee_pct),
                                "Fee_Cash": fee,
                                "Net_Proceeds": net,
                                "Basis_Type": tp_basis,
                                "Basis_Price_Used": float(basis_price),
                            })
                            last_tp_date = dt.date()

        # --- Buy logic (if base signal says 'consider buy') ---
        execute_buy = False
        if bool(base_signal.get(dt, False)):
            # monthly cap
            ym = (dt.year, dt.month)
            if (max_signals_per_month > 0) and (month_counts.get(ym, 0) >= max_signals_per_month):
                execute_buy = False
            else:
                # buy cooldown
                if (last_buy_date is None) or ((dt.date() - last_buy_date).days >= int(cooldown_days)):
                    # new high reset
                    if require_new_high_reset and (last_signal_rollmax_at_buy is not None):
                        if roll_max.loc[dt] > last_signal_rollmax_at_buy:
                            execute_buy = True
                        else:
                            execute_buy = False
                    else:
                        execute_buy = True

        # Trend filter
        if execute_buy and trend_filter == "Close > 200D SMA":
            execute_buy = bool(ind["Close"].loc[dt] > ind["SMA200"].loc[dt])
        elif execute_buy and trend_filter == "50D SMA > 200D SMA":
            execute_buy = bool(ind["SMA50"].loc[dt] > ind["SMA200"].loc[dt])

        # RSI filter
        if execute_buy and use_rsi:
            rsi_val = ind["RSI"].loc[dt]
            if pd.isna(rsi_val) or not (rsi_val <= float(rsi_max)):
                execute_buy = False

        # Allocation caps (pre-trade checks)
        if execute_buy:
            # cap by total invested (includes fees)
            total_invested_so_far = -sum(a for a in cashflows_amounts if a < 0)
            prospective_cash_out = float(buy_amount) * (1.0 + float(fee_pct) / 100.0)
            if (max_invested_usd > 0) and (total_invested_so_far + prospective_cash_out > float(max_invested_usd)):
                execute_buy = False

            # cap by position value
            position_value_now = current_units * float(close.loc[dt])
            if (max_position_value_usd > 0) and (position_value_now > float(max_position_value_usd)):
                execute_buy = False

        # Execute BUY
        if execute_buy:
            exec_price = float(close.loc[dt]) * (1.0 + float(slippage_pct) / 100.0)
            units_bought = float(buy_amount) / exec_price
            fee_cash = float(buy_amount) * (float(fee_pct) / 100.0)
            total_cash_out = float(buy_amount) + fee_cash

            # holdings & basis
            current_units += units_bought
            total_cost_excl_fees += float(buy_amount)  # fees excluded from basis
            last_buy_price = exec_price
            last_buy_date = dt.date()
            last_signal_rollmax_at_buy = roll_max.loc[dt]
            month_counts[ym] = month_counts.get(ym, 0) + 1

            # cash flow (negative)
            cashflows_dates.append(dt.date())
            cashflows_amounts.append(-total_cash_out)

            trades.append({
                "Type": "BUY",
                "Date": dt.date(),
                "Price_Close": float(close.loc[dt]),
                "Executed_Price": exec_price,
                "Units": units_bought,
                "USD_Spent_Excl_Fee": float(buy_amount),
                "Fee_%": float(fee_pct),
                "Fee_Cash": fee_cash,
                "Total_Cash_Out": total_cash_out,
                "Drawdown% (on signal)": float(ind["DrawdownPct"].loc[dt]),
                "RSI": float(ind["RSI"].loc[dt]) if not pd.isna(ind["RSI"].loc[dt]) else np.nan,
                "SMA50": float(ind["SMA50"].loc[dt]) if not pd.isna(ind["SMA50"].loc[dt]) else np.nan,
                "SMA200": float(ind["SMA200"].loc[dt]) if not pd.isna(ind["SMA200"].loc[dt]) else np.nan,
            })

        ind.loc[dt, "Units"] = current_units

    # Terminal value & metrics
    portfolio_value = ind["Units"] * ind["Close"]
    terminal_value = float(portfolio_value.iloc[-1])
    cashflows_dates.append(idx[-1].date())
    cashflows_amounts.append(terminal_value)

    try:
        irr = float(npf.xirr(cashflows_amounts, pd.to_datetime(cashflows_dates)))
    except Exception:
        irr = float("nan")

    total_invested = float(-sum(a for a in cashflows_amounts if a < 0))
    summary = {
        "initial_eth": float((DEFAULT_START_VALUE if start_value_usd is None else start_value_usd) / ref_price_usd),
        "final_eth": float(ind["Units"].iloc[-1]),
        "terminal_value": float(terminal_value),
        "total_invested": total_invested,  # includes starting value & all fees
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
st.caption("Educational tool. Data: Yahoo Finance via yfinance. Not financial advice.")

with st.sidebar:
    st.header("Backtest Period")
    start_date = st.date_input("Start date", value=DEFAULT_START_DATE)
    end_mode = st.radio(
        "End date mode",
        ["Rolling (use today's date)", "Fixed date"],
        index=0,
        help="Rolling updates daily; Fixed locks to your chosen cutoff."
    )
    if end_mode == "Rolling (use today's date)":
        end_date = date.today()
        st.info(f"Using today's date: {end_date.isoformat()}")
    else:
        end_date = st.date_input("Fixed end date", value=DEFAULT_FIXED_END_DATE, min_value=start_date)

    st.divider()
    st.header("Signal Logic")
    threshold_mode = st.selectbox(
        "Threshold mode",
        ["Fixed % (classic 5%)", "ATR multiple (vol-adaptive)"],
        index=0
    )
    if threshold_mode.startswith("ATR"):
        atr_period = st.number_input("ATR period (days)", min_value=5, max_value=100, value=DEFAULT_ATR_PERIOD, step=1)
        atr_mult = st.number_input("ATR multiple", min_value=0.5, max_value=5.0, value=DEFAULT_ATR_MULT, step=0.1)
        threshold_pct = DEFAULT_THRESHOLD_PCT
    else:
        atr_period = DEFAULT_ATR_PERIOD
        atr_mult = DEFAULT_ATR_MULT
        threshold_pct = st.number_input("Dip threshold (%)", min_value=1.0, max_value=50.0, value=DEFAULT_THRESHOLD_PCT, step=0.5)

    window_days = st.number_input("Week window (days)", min_value=2, max_value=30, value=DEFAULT_WINDOW_DAYS, step=1)

    st.divider()
    st.header("Filters / Risk Controls")
    trend_filter = st.selectbox("Trend filter", ["None", "Close > 200D SMA", "50D SMA > 200D SMA"], index=0)
    use_rsi = st.checkbox("Use RSI confirmation (buy only if RSI ≤ threshold)", value=DEFAULT_RSI_USE)
    rsi_period = st.number_input("RSI period", min_value=5, max_value=100, value=DEFAULT_RSI_PERIOD, step=1)
    rsi_max = st.number_input("RSI threshold (≤)", min_value=5.0, max_value=60.0, value=DEFAULT_RSI_MAX, step=1.0)
    require_new_high_reset = st.checkbox(
        "Require NEW rolling high after a buy before next signal",
        value=DEFAULT_REQUIRE_NEW_HIGH_RESET
    )
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
    max_invested_usd = st.number_input("Max total invested USD (0 = no cap)", min_value=0.0, max_value=10_000_000.0, value=DEFAULT_MAX_INVESTED_USD, step=100.0,
                                       help="Includes starting value and all buys with fees.")
    max_position_value_usd = st.number_input("Max position value USD (0 = no cap)", min_value=0.0, max_value=10_000_000.0, value=DEFAULT_MAX_POSITION_VALUE_USD, step=100.0,
                                             help="If units × price exceeds this, new buys are skipped.")

    st.divider()
    st.header("Starting Portfolio")
    start_value_usd = st.number_input("Starting value (USD)", min_value=0.0, value=DEFAULT_START_VALUE, step=100.0)
    ref_price_usd = st.number_input(
        "Reference price to back into ETH units",
        min_value=0.01,
        value=DEFAULT_REF_PRICE,
        step=0.01,
        help="Initial ETH units = Starting value ÷ Reference price. Your instruction: $2,500 ÷ $354.31.",
    )

# Fetch & prep data
ohlc = fetch_ohlc(start_date, end_date)
if ohlc.empty:
    st.error("No price data returned for the selected dates.")
    st.stop()

ind = compute_indicators(ohlc, rsi_period=int(rsi_period), atr_period=int(atr_period))

# Base dip signals
base_signal = compute_base_signals(
    ind,
    threshold_mode=threshold_mode,
    fixed_pct=float(threshold_pct if threshold_mode.startswith("Fixed") else DEFAULT_THRESHOLD_PCT),
    atr_mult=float(atr_mult)
)

# Run simulation
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
        st.download_button(
            "Download trades (CSV)",
            data=trades_df.to_csv(index=False),
            file_name="trades.csv",
            mime="text/csv",
        )
    else:
        st.info("No trades executed in the selected window.")

with right:
    st.markdown("### Cash Flows (for XIRR)")
    cf = details["cashflows"].copy()
    st.dataframe(cf, use_container_width=True, hide_index=True)
    st.download_button(
        "Download cash flows (CSV)",
        data=cf.to_csv(index=False),
        file_name="cashflows.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    """
**Methodology & Controls**
- **Dip buy:** First cross where drawdown vs rolling 7-day high ≥ threshold (fixed 5% or ATR×mult).
- **Filters:** Trend (SMA), RSI, new-high reset, buy cooldown, monthly cap.
- **Execution:** Slippage & fees on both buys and sells. Average-cost accounting for basis.
- **Take-profit:** When price ≥ basis × (1 + TP%), sell % of holdings (with its own cooldown). Basis can be **Average cost** or **Last buy price**.
- **Caps:** Skip buys if total invested (incl. starting & fees) would exceed cap, or if position value already exceeds cap.
- **Performance:** XIRR from cash flows (starting outflow, all net buys/sells, terminal value inflow).
"""
)
