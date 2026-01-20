# app.py
import streamlit as st
import pandas as pd

from scheme_finder import render_scheme_finder
from portfolio_calc import (
    DEFAULT_SETTINGS,
    compute_etf_portfolio,
    compute_mf_portfolio,
)

# MUST be called once, at the top
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("Portfolio Analyzer (Daily)")

# ---- Page navigation FIRST ----
page = st.sidebar.radio("Page", ["Portfolio Analyzer", "Scheme Code Finder"])

if page == "Scheme Code Finder":
    render_scheme_finder()
    st.stop()

# -----------------------------
# Analyzer page starts here
# -----------------------------
st.subheader("Choose portfolio type")
portfolio_type = st.radio(
    "Select one:",
    ["ETF Portfolio", "Mutual Fund Portfolio"],
    horizontal=True
)

st.sidebar.header("Global Settings (Daily)")
settings = DEFAULT_SETTINGS.copy()

# Daily defaults already in DEFAULT_SETTINGS, but allow override
settings["period"] = st.sidebar.text_input("period", value=settings["period"])
settings["interval"] = st.sidebar.selectbox("interval", ["1d", "1wk"], index=0)

# Optional: auto-adjust annualization_factor based on interval selection
default_ann = 252 if settings["interval"] == "1d" else 52
settings["annualization_factor"] = st.sidebar.number_input(
    "annualization_factor",
    value=int(settings.get("annualization_factor", default_ann)),
    min_value=1
)

settings["market_ticker"] = st.sidebar.text_input("market_ticker", value=settings["market_ticker"])

settings["rf_annual"] = st.sidebar.number_input(
    "rf_annual (decimal)", value=float(settings["rf_annual"]), min_value=0.0, step=0.005
)
settings["use_excess_returns_for_beta"] = st.sidebar.checkbox(
    "Use excess returns for beta (CAPM)", value=bool(settings["use_excess_returns_for_beta"])
)

use_log_returns = st.sidebar.checkbox("Use log returns", value=False)

st.divider()

# -----------------------------
# Input tables
# -----------------------------
if portfolio_type == "ETF Portfolio":
    st.subheader("ETF Inputs")
    st.caption("Enter ETF tickers and invested amount per ETF. (Total is computed automatically.)")

    default_rows = [
        {"ETF Ticker": "NIFTYBEES.NS", "Invested": 254224},
        {"ETF Ticker": "SILVERIETF.NS", "Invested": 63175},
        {"ETF Ticker": "GOLDBEES.NS", "Invested": 91636},
        {"ETF Ticker": "BANKBEES.NS", "Invested": 75419},
    ]

    edited = st.data_editor(
        pd.DataFrame(default_rows),
        num_rows="dynamic",
        use_container_width=True
    )

    edited = edited.dropna()
    edited["ETF Ticker"] = edited["ETF Ticker"].astype(str).str.strip()
    edited = edited[edited["ETF Ticker"] != ""]
    edited["Invested"] = pd.to_numeric(edited["Invested"], errors="coerce")
    edited = edited.dropna(subset=["Invested"])

    tickers = edited["ETF Ticker"].tolist()
    invested_amounts = dict(zip(edited["ETF Ticker"], edited["Invested"]))
    total_invested = float(edited["Invested"].sum()) if not edited.empty else 0.0

    st.info(f"Total Investment (₹): {total_invested:,.0f}")
    run_label = "Compute ETF Portfolio"

else:
    st.subheader("Mutual Fund Inputs")
    st.caption("Enter MF scheme codes and invested amount per scheme. NAV history will be fetched from mfapi.in.")

    default_rows = [
        {"Scheme Code": "145075", "Invested": 100000},
        {"Scheme Code": "120821", "Invested": 100000},
        {"Scheme Code": "113177", "Invested": 100000},
    ]

    edited = st.data_editor(
        pd.DataFrame(default_rows),
        num_rows="dynamic",
        use_container_width=True
    )

    edited = edited.dropna()
    edited["Scheme Code"] = edited["Scheme Code"].astype(str).str.strip()
    edited = edited[edited["Scheme Code"] != ""]
    edited["Invested"] = pd.to_numeric(edited["Invested"], errors="coerce")
    edited = edited.dropna(subset=["Invested"])

    scheme_codes = edited["Scheme Code"].tolist()
    invested_amounts = dict(zip(edited["Scheme Code"], edited["Invested"]))
    total_invested = float(edited["Invested"].sum()) if not edited.empty else 0.0

    st.info(f"Total Investment (₹): {total_invested:,.0f}")
    run_label = "Compute Mutual Fund Portfolio"

st.divider()
st.subheader("Run Analysis")

# -----------------------------
# Compute + display
# -----------------------------
if st.button(run_label):
    try:
        if portfolio_type == "ETF Portfolio":
            if not tickers:
                raise ValueError("Please enter at least one ETF ticker.")
            res = compute_etf_portfolio(
                tickers=tickers,
                invested_amounts=invested_amounts,
                settings=settings,
                use_log_returns=use_log_returns,
            )
        else:
            if not scheme_codes:
                raise ValueError("Please enter at least one MF scheme code.")
            res = compute_mf_portfolio(
                scheme_codes=scheme_codes,
                invested_amounts=invested_amounts,
                settings=settings,
                use_log_returns=use_log_returns,
            )

        # KPI HEADER
        st.subheader("KPIs")
        k1, k2, k3, k4, k5, k6 = st.columns(6)

        k1.metric("Total Invested (₹)", f"{res['total_invested']:,.0f}")
        k2.metric("Variance (Period)", f"{res['var_period']:.10f}")
        k3.metric("Volatility (Period)", f"{res['vol_period']:.4%}")
        k4.metric("Volatility (Annual)", f"{res['vol_annual']:.4%}")
        k5.metric("Portfolio Beta", f"{res['beta_port_weighted']:.3f}")
        k6.metric("Sharpe (Annual)", f"{res['sharpe_annual']:.3f}")

        k7, k8, k9 = st.columns(3)
        k7.metric("Treynor (Annual)", f"{res['treynor_annual']:.4f}")
        k8.metric("M² (Annual)", f"{res['m2_annual']:.4%}")
        k9.metric("RF per Period", f"{res['rf_period']:.6f}")

        st.success(
            f"✅ Done. Beta vs {res['market_ticker']} = **{res['beta_port_weighted']:.3f}** | "
            f"Annual Vol = **{res['vol_annual']:.2%}** | Sharpe(ann) = **{res['sharpe_annual']:.3f}**"
        )

        st.divider()

        # DETAILS
        st.subheader("Details")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Weights")
            st.dataframe((res["weights_named"] * 100).round(4).to_frame("Weight (%)"), use_container_width=True)

        with c2:
            st.markdown("### Individual Betas")
            st.dataframe(res["betas_named"].round(4).to_frame("Beta"), use_container_width=True)
            st.caption(f"Portfolio beta (from returns check): {res['beta_port_from_returns']:.4f}")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### Correlation (Period)")
            st.dataframe(res["corr"].round(4), use_container_width=True)

        with c4:
            st.markdown("### Covariance (Period)")
            st.dataframe(res["cov_period"].round(10), use_container_width=True)

        st.markdown("### Recent returns (last 10 periods)")
        st.dataframe(res["recent_returns"].round(6), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")