# app.py
import streamlit as st
import pandas as pd

from portfolio_calc import (
    load_portfolio_json,
    save_portfolio_json,
    compute_portfolio_stats,
    DEFAULT_SETTINGS
)

JSON_PATH = "portfolio.json"

st.set_page_config(page_title="Portfolio Variance", layout="wide")
st.title("Portfolio Variance Calculator")

portfolio = load_portfolio_json(JSON_PATH)

st.sidebar.header("Portfolio Source")
use_json = portfolio is not None

if use_json:
    st.sidebar.success(f"Found {JSON_PATH} — using it.")
    if st.sidebar.button("Ignore JSON and enter manually"):
        portfolio = None
        use_json = False
else:
    st.sidebar.warning(f"No {JSON_PATH} found — please enter manually.")

st.sidebar.header("Settings")
settings = DEFAULT_SETTINGS.copy()
if portfolio and "data_settings" in portfolio:
    settings.update(portfolio["data_settings"])

settings["period"] = st.sidebar.text_input("period", value=settings["period"])
settings["interval"] = st.sidebar.text_input("interval", value=settings["interval"])
settings["price_field"] = st.sidebar.selectbox(
    "price_field", ["Adj Close", "Close", "Open", "High", "Low"],
    index=["Adj Close", "Close", "Open", "High", "Low"].index(settings["price_field"])
)
settings["annualization_factor"] = st.sidebar.number_input(
    "annualization_factor", value=int(settings["annualization_factor"]), min_value=1
)

use_log_returns = st.sidebar.checkbox("Use log returns", value=False)

if not use_json:
    st.subheader("Manual Input")
    default_rows = [
        {"Ticker": "NIFTYBEES.NS", "Invested": 254224},
        {"Ticker": "SILVERIETF.NS", "Invested": 63175},
        {"Ticker": "GOLDBEES.NS", "Invested": 91636},
        {"Ticker": "BANKBEES.NS", "Invested": 75419},
    ]

    edited = st.data_editor(
        pd.DataFrame(default_rows),
        num_rows="dynamic",
        use_container_width=True
    )

    edited = edited.dropna()
    edited["Ticker"] = edited["Ticker"].astype(str).str.strip()
    edited = edited[edited["Ticker"] != ""]
    edited["Invested"] = pd.to_numeric(edited["Invested"], errors="coerce")
    edited = edited.dropna(subset=["Invested"])

    assets = edited["Ticker"].tolist()
    invested_amounts = dict(zip(edited["Ticker"], edited["Invested"]))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Save to portfolio.json"):
            save_portfolio_json(JSON_PATH, assets, invested_amounts, settings)
            st.success(f"Saved {JSON_PATH}. Refresh to auto-load.")
    with col_b:
        st.caption("You can compute without saving too.")
else:
    assets = portfolio["assets"]
    invested_amounts = portfolio["invested_amounts"]

st.divider()
st.subheader("Run")

if st.button("Compute portfolio variance"):
    try:
        res = compute_portfolio_stats(
            assets=assets,
            invested_amounts=invested_amounts,
            data_settings=settings,
            use_log_returns=use_log_returns
        )

        # =========================
        # KPI HEADER (TOP)
        # =========================
        st.subheader("KPIs")
        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric("Total Invested (₹)", f"{res['total_invested']:,.0f}")
        k2.metric("Variance (Weekly)", f"{res['var_weekly']:.10f}")
        k3.metric("Variance (Annual)", f"{res['var_annual']:.10f}")
        k4.metric("Volatility (Weekly)", f"{res['vol_weekly']:.4%}")
        k5.metric("Volatility (Annual)", f"{res['vol_annual']:.4%}")

        st.success(
            f"✅ Your portfolio variance is **{res['var_annual']:.10f} (annual)** "
            f"and **{res['var_weekly']:.10f} (weekly)**"
        )

        st.divider()

        # =========================
        # DETAILS BELOW KPIs
        # =========================
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Weights")
            st.dataframe(
                (res["weights"] * 100).round(4).to_frame("Weight (%)"),
                use_container_width=True
            )

        with c2:
            st.markdown("### Correlation (Weekly)")
            st.dataframe(res["corr"].round(4), use_container_width=True)

        st.markdown("### Covariance (Weekly)")
        st.dataframe(res["cov_weekly"].round(10), use_container_width=True)

        st.markdown("### Recent returns (last 10 weeks)")
        st.dataframe(res["returns"].tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
