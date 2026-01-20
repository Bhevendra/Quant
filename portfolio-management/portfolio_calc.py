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
    "annualization_factor": 52
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

def compute_portfolio_stats(
    assets: list[str],
    invested_amounts: dict,
    data_settings: dict | None = None,
    use_log_returns: bool = False
) -> dict:
    """
    Computes weights, covariance matrices, portfolio variance & volatility.
    Returns a dict with results.
    """
    settings = DEFAULT_SETTINGS.copy()
    if data_settings:
        settings.update(data_settings)

    # Validate tickers and amounts
    if not assets:
        raise ValueError("No assets provided.")
    for t in assets:
        if t not in invested_amounts:
            raise ValueError(f"Missing invested amount for {t}.")
    if any(float(invested_amounts[t]) < 0 for t in assets):
        raise ValueError("Invested amounts must be non-negative.")

    # Prices & covariance
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

    # Weights
    amounts = pd.Series({t: float(invested_amounts[t]) for t in assets})
    total = float(amounts.sum())
    if total <= 0:
        raise ValueError("Total invested must be > 0.")

    w = (amounts / total).reindex(assets)

    # Align covariance order
    Sigma_w = cov_weekly.reindex(index=assets, columns=assets)
    Sigma_a = cov_annual.reindex(index=assets, columns=assets)

    # Portfolio variance & vol
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
        "cov_weekly": cov_weekly,
        "cov_annual": cov_annual,
        "corr": corr,
        "var_weekly": var_weekly,
        "vol_weekly": vol_weekly,
        "var_annual": var_annual,
        "vol_annual": vol_annual
    }
