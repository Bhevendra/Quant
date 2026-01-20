# Portfolio Analyzer (ETF + Mutual Funds) — Streamlit App

A simple, practical portfolio analytics app that lets you:
- Analyze an **ETF portfolio** (via Yahoo Finance)
- Analyze a **Mutual Fund portfolio** (via mfapi.in NAV history)
- View key risk/return metrics like **Variance, Volatility, Beta, Sharpe, Treynor, and M²**
- Explore portfolio **weights, individual betas, covariance/correlation**, and **recent returns**
- Find **Mutual Fund scheme codes** using a built-in scheme search page

This is designed for anyone who wants quick, transparent portfolio analytics without manually building spreadsheets.

---

## What this app helps you do

### ✅ ETF Portfolio
- Enter ETF tickers (example: `NIFTYBEES.NS`, `GOLDBEES.NS`) and invested amounts.
- The app downloads **daily adjusted close prices** from Yahoo Finance (via `yfinance`).
- It computes daily returns, risk metrics, and performance ratios.

### ✅ Mutual Fund Portfolio
- Enter MF scheme codes (example: `145075`) and invested amounts.
- The app fetches **daily NAV history** from `https://api.mfapi.in/mf/<scheme_code>`.
- It aligns MF NAV dates with market dates and computes the same metrics.

### ✅ Scheme Code Finder
- Search by mutual fund name (partial or full).
- The app pulls scheme master list from `https://api.mfapi.in/mf` and returns matching scheme codes.

---

## Metrics you get (at the top as KPIs)

- **Variance (Period)**: variance of portfolio returns at the chosen interval (daily/weekly)
- **Volatility (Period)**: standard deviation of portfolio returns
- **Volatility (Annual)**: annualized volatility (252 for daily, 52 for weekly)
- **Portfolio Beta**: market sensitivity vs your selected market index (default: `^NSEI`)
- **Sharpe Ratio**: risk-adjusted return using risk-free rate
- **Treynor Ratio**: return per unit of market risk (beta)
- **M² (Modigliani–Modigliani)**: Sharpe translated into “return units” at benchmark risk

Below the KPIs you’ll also see:
- **Weights**
- **Individual asset/scheme betas**
- **Correlation matrix**
- **Covariance matrix**
- **Recent returns (last 10 periods)**

---

## How beta is calculated (simple explanation)

Beta is computed using returns:

\[
\beta = \frac{cov(r_p, r_m)}{var(r_m)}
\]

Where:
- \(r_p\) = portfolio returns
- \(r_m\) = market returns (default: NIFTY 50 / `^NSEI`)

You can optionally compute beta using **excess returns** (CAPM style):
- \(r_p - r_f\) and \(r_m - r_f\)

> Note: subtracting a constant risk-free rate typically **does not change beta** (covariance/variance are unchanged), but it’s included for CAPM correctness and consistency.

---

## Project structure

Typical layout:

```
portfolio-management/
app.py                 # Streamlit UI (pages + inputs + tables + KPI display)
portfolio_calc.py      # Analytics engine (returns, cov, beta, sharpe, treynor, M²)
scheme_finder.py       # Scheme Code Finder page + mfapi master search logic
requirements.txt
.streamlit/
config.toml          # Streamlit config (no forced port for cloud)
README.md
runtime.txt              # Python runtime pin for Streamlit Cloud (repo root)
```


------
------
```
---

## Run locally (or in GitHub Codespaces)

### 1) Install dependencies
```bash
pip install -r portfolio-management/requirements.txt
streamlit run portfolio-management/app.py
cd portfolio-management
streamlit run app.py
```
