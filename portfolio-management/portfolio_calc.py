
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# Daily defaults now
DEFAULT_SETTINGS = {
    "period": "5y",
    "interval": "1d",
    "price_field": "Adj Close",
    "annualization_factor": 252,     # daily -> ~252 trading days
    "market_ticker": "^NSEI",        # NIFTY 50
    "rf_annual": 0.0,                # e.g. 0.07 for 7%
    "use_excess_returns_for_beta": False,
}


# -----------------------------
# JSON helpers (optional)
# -----------------------------
def load_portfolio_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_portfolio_json(path: str, payload: dict) -> dict:
    payload = dict(payload)
    payload["as_of"] = datetime.now().strftime("%Y-%m-%d")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


# -----------------------------
# Data loaders
# -----------------------------
def download_prices_yf(
    tickers: List[str],
    period: str,
    interval: str,
    price_field: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a wide price table: index=Date, columns=tickers
    Uses period/interval for ETFs; for MF market alignment we can use start/end.
    """
    if start or end:
        df = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
    else:
        df = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )

    prices = pd.DataFrame({t: df[t][price_field] for t in tickers}).sort_index()
    prices = prices.dropna(how="any")
    if prices.empty:
        raise ValueError("No price data available after cleaning. Check tickers/period/interval.")
    return prices


def download_mf_nav_prices(scheme_codes: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Downloads NAV history for each MF scheme code using mfapi.in
    Returns:
      - prices: wide NAV table, index=date, columns=scheme_code (as string)
      - meta: dict scheme_code -> scheme_name
    """
    series_list = []
    meta = {}

    for code in scheme_codes:
        code_str = str(code).strip()
        url = f"https://api.mfapi.in/mf/{code_str}"
        data = requests.get(url, timeout=30).json()

        scheme_name = data.get("meta", {}).get("scheme_name", code_str)
        meta[code_str] = scheme_name

        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df["nav"] = df["nav"].astype(float)
        df = df.sort_values("date")

        s = df.set_index("date")["nav"].rename(code_str)
        series_list.append(s)

    prices = pd.concat(series_list, axis=1).sort_index()

    # Keep only dates where all NAVs exist (simple + consistent)
    prices = prices.dropna(how="any")
    if prices.empty:
        raise ValueError("No mutual fund NAV data after cleaning. Check scheme codes.")
    return prices, meta


# -----------------------------
# Core math
# -----------------------------
def compute_returns(prices: pd.DataFrame, use_log_returns: bool = False) -> pd.DataFrame:
    if use_log_returns:
        rets = np.log(prices / prices.shift(1)).dropna()
    else:
        rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("Returns are empty. Not enough data points.")
    return rets


def compute_betas(asset_returns: pd.DataFrame, market_returns: pd.Series) -> pd.Series:
    aligned = asset_returns.join(market_returns.rename("MKT"), how="inner").dropna()
    mkt = aligned["MKT"]
    var_m = float(mkt.var(ddof=1))
    if var_m == 0:
        raise ValueError("Market return variance is zero; cannot compute beta.")

    betas = {}
    for col in asset_returns.columns:
        betas[col] = float(aligned[col].cov(mkt) / var_m)
    return pd.Series(betas)


def annual_rf_to_period_rf(rf_annual: float, annualization_factor: float) -> float:
    # Compound-consistent conversion
    return (1.0 + rf_annual) ** (1.0 / annualization_factor) - 1.0


def compute_stats_from_prices(
    prices: pd.DataFrame,
    invested_amounts: Dict[str, float],
    settings: dict,
    market_prices: pd.DataFrame,
    use_log_returns: bool = False,
    label_map: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Generic engine: given a price table (ETF or MF), compute:
      - weights, cov, corr
      - individual betas + portfolio beta
      - variance/volatility (period + annual)
      - Sharpe, Treynor, M2 (period + annualized)
      - recent returns
    """
    assets = list(prices.columns)
    ann_factor = float(settings["annualization_factor"])

    # weights
    amounts = pd.Series({a: float(invested_amounts[a]) for a in assets}).reindex(assets)
    total = float(amounts.sum())
    if total <= 0:
        raise ValueError("Total invested must be > 0.")
    w = (amounts / total)

    # returns (asset)
    rets = compute_returns(prices, use_log_returns=use_log_returns)

    # returns (market)
    mkt_ticker = settings["market_ticker"]
    mkt_series = market_prices[mkt_ticker]
    if use_log_returns:
        mkt_rets = np.log(mkt_series / mkt_series.shift(1)).dropna()
    else:
        mkt_rets = mkt_series.pct_change().dropna()

    # align on dates
    aligned = rets.join(mkt_rets.rename("MKT"), how="inner").dropna()
    rets_aligned = aligned[assets]
    mkt_aligned = aligned["MKT"]

    # covariance/correlation (period)
    cov_period = rets_aligned.cov()
    cov_annual = cov_period * ann_factor
    corr = rets_aligned.corr()

    # risk-free handling (for Sharpe/Treynor, and optionally for beta)
    rf_annual = float(settings.get("rf_annual", 0.0))
    rf_period = annual_rf_to_period_rf(rf_annual, ann_factor)

    use_excess_for_beta = bool(settings.get("use_excess_returns_for_beta", False))
    if use_excess_for_beta:
        rets_for_beta = rets_aligned.sub(rf_period)
        mkt_for_beta = mkt_aligned.sub(rf_period)
    else:
        rets_for_beta = rets_aligned
        mkt_for_beta = mkt_aligned

    # betas
    betas = compute_betas(rets_for_beta, mkt_for_beta)
    beta_port_weighted = float((w * betas.reindex(assets)).sum())

    port_rets_for_beta = (rets_for_beta * w.values).sum(axis=1)
    beta_port_from_returns = float(
        port_rets_for_beta.cov(mkt_for_beta) / mkt_for_beta.var(ddof=1)
    )

    # portfolio returns (non-excess)
    portfolio_returns = (rets_aligned * w.values).sum(axis=1)

    # variance/vol
    Sigma = cov_period.reindex(index=assets, columns=assets)
    var_period = float(w.T @ Sigma @ w)
    vol_period = float(np.sqrt(var_period))

    var_annual = var_period * ann_factor
    vol_annual = float(np.sqrt(var_annual))

    # Sharpe / Treynor / M2
    # Use excess returns for performance ratios (standard definition)
    rp_excess = portfolio_returns - rf_period
    mean_excess_period = float(rp_excess.mean())

    sharpe_period = mean_excess_period / vol_period
    sharpe_annual = sharpe_period * np.sqrt(ann_factor)

    treynor_period = mean_excess_period / beta_port_weighted
    treynor_annual = (mean_excess_period * ann_factor) / beta_port_weighted

    sigma_m_period = float(mkt_aligned.std(ddof=1))
    m2_period = rf_period + sharpe_period * sigma_m_period
    m2_annual = (1.0 + m2_period) ** ann_factor - 1.0

    # recent returns
    recent_returns = rets_aligned.tail(10)

    # label mapping (MF scheme_code -> scheme_name)
    if label_map is None:
        label_map = {a: a for a in assets}

    betas_named = betas.rename(index=label_map)
    weights_named = w.rename(index=label_map)

    # Also provide "raw id" order for joins
    return {
        "settings": settings,
        "assets": assets,
        "labels": label_map,

        "total_invested": total,
        "weights": w,
        "weights_named": weights_named,

        "prices": prices,
        "returns": rets_aligned,
        "portfolio_returns": portfolio_returns,

        "market_ticker": mkt_ticker,
        "market_returns": mkt_aligned,

        "cov_period": cov_period,
        "cov_annual": cov_annual,
        "corr": corr,

        "rf_period": rf_period,
        "betas": betas,
        "betas_named": betas_named,
        "beta_port_weighted": beta_port_weighted,
        "beta_port_from_returns": beta_port_from_returns,

        "var_period": var_period,
        "vol_period": vol_period,
        "var_annual": var_annual,
        "vol_annual": vol_annual,

        "sharpe_period": sharpe_period,
        "sharpe_annual": sharpe_annual,
        "treynor_period": treynor_period,
        "treynor_annual": treynor_annual,
        "m2_period": m2_period,
        "m2_annual": m2_annual,

        "recent_returns": recent_returns,
    }


# -----------------------------
# Public entry points
# -----------------------------
def compute_etf_portfolio(
    tickers: List[str],
    invested_amounts: Dict[str, float],
    settings: dict,
    use_log_returns: bool = False,
) -> dict:
    prices = download_prices_yf(
        tickers=tickers,
        period=settings["period"],
        interval=settings["interval"],
        price_field=settings["price_field"],
    )

    # market prices aligned to ETF price date range
    start = prices.index.min().strftime("%Y-%m-%d")
    end = (prices.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    market_prices = download_prices_yf(
        tickers=[settings["market_ticker"]],
        period=settings["period"],
        interval=settings["interval"],
        price_field=settings["price_field"],
        start=start,
        end=end,
    )

    label_map = {t: t for t in tickers}
    return compute_stats_from_prices(
        prices=prices,
        invested_amounts=invested_amounts,
        settings=settings,
        market_prices=market_prices,
        use_log_returns=use_log_returns,
        label_map=label_map,
    )


def compute_mf_portfolio(
    scheme_codes: List[str],
    invested_amounts: Dict[str, float],
    settings: dict,
    use_log_returns: bool = False,
) -> dict:
    prices, meta = download_mf_nav_prices(scheme_codes)

    # market prices aligned to MF NAV date range
    start = prices.index.min().strftime("%Y-%m-%d")
    end = (prices.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    market_prices = download_prices_yf(
        tickers=[settings["market_ticker"]],
        period=settings["period"],
        interval=settings["interval"],
        price_field=settings["price_field"],
        start=start,
        end=end,
    )

    # label_map: scheme_code -> scheme_name
    label_map = {code: meta.get(code, code) for code in scheme_codes}
    return compute_stats_from_prices(
        prices=prices,
        invested_amounts=invested_amounts,
        settings=settings,
        market_prices=market_prices,
        use_log_returns=use_log_returns,
        label_map=label_map,
    )