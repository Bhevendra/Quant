# portfolio_calc.py
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_SETTINGS = {
    "period": "5y",
    "interval": "1wk",
    "price_field": "Adj Close",
    "annualization_factor": 52,
    "market_ticker": "^NSEI",       # NIFTY 50 index on yfinance
    "rf_annual": 0.0,               # optional risk-free annual rate (decimal). Ex: 0.07 for 7%
    "use_excess_returns_for_beta": False
}


def load_portfolio_json(path: str) -> dict | None:
    """Load portfolio.json if it exists, else return None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_portfolio_json(path: str, assets: list[str], invested_amounts: dict, data_settings: dict) -> dict:
    """Save a portfolio payload to JSON and return the payload."""
    payload = {
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "assets": assets,
        "invested_amounts": invested_amounts,
        "total_invested": float(sum(invested_amounts.values())),
        "data_settings": data_settings
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def download_prices(assets: list[str], period: str, interval: str, price_field: str) -> pd.DataFrame:
    """Download prices from yfinance and return Date x Ticker price table."""
    df = yf.download(
        tickers=assets,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    # Build wide price table
    prices = pd.DataFrame({t: df[t][price_field] for t in assets}).sort_index()

    # Clean: keep only rows where all tickers have data
    prices = prices.dropna(how="any")
    if prices.empty:
        raise ValueError("No price data available after cleaning. Check tickers or interval/period.")
    return prices


def compute_covariance(prices: pd.DataFrame, use_log_returns: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (returns_df, cov_weekly)."""
    if use_log_returns:
        rets = np.log(prices / prices.shift(1)).dropna()
    else:
        rets = prices.pct_change().dropna()

    if rets.empty:
        raise ValueError("Returns are empty. Not enough data points.")
    cov_weekly = rets.cov()
    return rets, cov_weekly


def compute_betas(asset_returns: pd.DataFrame, market_returns: pd.Series) -> pd.Series:
    """
    Beta_i = cov(r_i, r_m) / var(r_m)

    asset_returns: DataFrame with columns = assets
    market_returns: Series
    """
    aligned = asset_returns.join(market_returns.rename("MKT"), how="inner")
    mkt = aligned["MKT"]
    var_m = float(mkt.var(ddof=1))
    if var_m == 0:
        raise ValueError("Market return variance is zero; cannot compute beta.")

    betas = {}
    for col in asset_returns.columns:
        cov_im = float(aligned[col].cov(mkt))
        betas[col] = cov_im / var_m

    return pd.Series(betas)


def compute_portfolio_stats(
    assets: list[str],
    invested_amounts: dict,
    data_settings: dict | None = None,
    use_log_returns: bool = False
) -> dict:
    """
    Computes:
      - weights
      - covariance matrices (weekly + annualized)
      - portfolio variance & volatility (weekly + annual)
      - individual asset betas vs market
      - portfolio beta (weighted avg + from portfolio returns sanity-check)

    Returns a dict of results.
    """
    settings = DEFAULT_SETTINGS.copy()
    if data_settings:
        settings.update(data_settings)

    # Validate
    if not assets:
        raise ValueError("No assets provided.")
    for t in assets:
        if t not in invested_amounts:
            raise ValueError(f"Missing invested amount for {t}.")
    if any(float(invested_amounts[t]) < 0 for t in assets):
        raise ValueError("Invested amounts must be non-negative.")

    # ---- Weights FIRST (needed for portfolio beta/returns) ----
    amounts = pd.Series({t: float(invested_amounts[t]) for t in assets})
    total = float(amounts.sum())
    if total <= 0:
        raise ValueError("Total invested must be > 0.")
    w = (amounts / total).reindex(assets)

    # ---- Prices & covariance for assets ----
    prices = download_prices(
        assets=assets,
        period=settings["period"],
        interval=settings["interval"],
        price_field=settings["price_field"]
    )
    rets, cov_weekly = compute_covariance(prices, use_log_returns=use_log_returns)
    corr = rets.corr()

    # Annualize covariance
    ann_factor = float(settings["annualization_factor"])
    cov_annual = cov_weekly * ann_factor

    # ---- Market data for beta ----
    mkt_ticker = settings.get("market_ticker", "^NSEI")
    mkt_prices = download_prices(
        assets=[mkt_ticker],
        period=settings["period"],
        interval=settings["interval"],
        price_field=settings["price_field"]
    )

    mkt_series = mkt_prices[mkt_ticker]
    if use_log_returns:
        mkt_rets = np.log(mkt_series / mkt_series.shift(1)).dropna()
    else:
        mkt_rets = mkt_series.pct_change().dropna()

    # Optional: excess returns beta (CAPM style)
    if settings.get("use_excess_returns_for_beta", False):
        rf_annual = float(settings.get("rf_annual", 0.0))
        rf_period = (1.0 + rf_annual) ** (1.0 / ann_factor) - 1.0  # convert annual rf to weekly rf
        rets_for_beta = rets.sub(rf_period)
        mkt_for_beta = mkt_rets.sub(rf_period)
    else:
        rets_for_beta = rets
        mkt_for_beta = mkt_rets

    # Align asset returns columns to 'assets' order
    rets_for_beta = rets_for_beta.reindex(columns=assets)

    # ---- Betas ----
    betas = compute_betas(rets_for_beta, mkt_for_beta)

    # Portfolio beta (method 1): weighted avg of asset betas
    beta_port_weighted = float((w * betas.reindex(assets)).sum())

    # Portfolio beta (method 2): beta from portfolio returns vs market (sanity check)
    port_rets_for_beta = (rets_for_beta[assets] * w.values).sum(axis=1)
    aligned_pm = port_rets_for_beta.to_frame("PORT").join(mkt_for_beta.rename("MKT"), how="inner")
    beta_port_from_returns = float(aligned_pm["PORT"].cov(aligned_pm["MKT"]) / aligned_pm["MKT"].var(ddof=1))

    # Portfolio returns (non-excess; for display)
    portfolio_returns = (rets[assets] * w.values).sum(axis=1)

    # ---- Variance & vol ----
    Sigma_w = cov_weekly.reindex(index=assets, columns=assets)
    Sigma_a = cov_annual.reindex(index=assets, columns=assets)

    var_weekly = float(w.T @ Sigma_w @ w)
    var_annual = float(w.T @ Sigma_a @ w)
    vol_weekly = float(np.sqrt(var_weekly))
    vol_annual = float(np.sqrt(var_annual))

    return {
        "settings": settings,
        "prices": prices,
        "returns": rets,
        "weights": w,
        "total_invested": total,

        "market_ticker": mkt_ticker,
        "market_returns": mkt_rets,

        "betas": betas,
        "beta_port_weighted": beta_port_weighted,
        "beta_port_from_returns": beta_port_from_returns,

        "portfolio_returns": portfolio_returns,

        "cov_weekly": cov_weekly,
        "cov_annual": cov_annual,
        "corr": corr,

        "var_weekly": var_weekly,
        "vol_weekly": vol_weekly,
        "var_annual": var_annual,
        "vol_annual": vol_annual
    }
