# ETH Buy-the-Dip — Streamlit App

A free Streamlit app that backtests and tracks an **ETH buy-on-dips** strategy starting from **Sep 27, 2020** with an initial purchase of **$2,500 at $354.31/ETH**.  
Signals fire when price drops **≥ threshold vs the rolling 7-day high** (classic 5% or volatility-adaptive with ATR). The app simulates buys/sells, fees/slippage, and computes **XIRR** from cash flows.

> Educational tool only — **not** financial advice. Data: Yahoo Finance via `yfinance`.

---

## Quickstart (Streamlit Community Cloud — Free)

1. Create a **public GitHub repo** and add three files:
   - `app.py`
   - `requirements.txt` (this file)
   - `README.md` (this file)

2. Go to **https://share.streamlit.io**  
   Sign in with GitHub → **New app** → pick your repo → set **`app.py`** as the main file → **Deploy**.

That’s it — you’ll get a public URL for the app.

---

## Run locally

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
