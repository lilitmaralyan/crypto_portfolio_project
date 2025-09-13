import pandas as pd
import numpy as np

def compute_equal_weights(monthly_returns: pd.DataFrame) -> pd.Series:
    """
    Computes equal weights for a given month.

    Parameters
    ----------
    monthly_returns : pd.DataFrame
        Daily returns for one month (coins as columns, dates as index)

    Returns
    -------
    pd.Series
        Weights for each coin (sums to 1)
    """
    n_assets = monthly_returns.shape[1]
    weights = pd.Series(1/n_assets, index=monthly_returns.columns)
    return weights


def compute_vol_scaled_weights(prev_month_returns: pd.DataFrame) -> pd.Series:
    """
    Computes volatility-scaled weights based on the PREVIOUS month's returns.

    Parameters
    ----------
    prev_month_returns : pd.DataFrame
        Daily returns for the previous month (coins as columns, dates as index)

    Returns
    -------
    pd.Series
        Volatility-scaled weights for each coin (sums to 1)
    """
    vol = prev_month_returns.std()
    inv_vol = 1 / vol.replace(0, np.nan)  # avoid div by zero
    weights = inv_vol / inv_vol.sum()
    return weights

def apply_weights(month_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Applies weights to a month's daily returns to compute portfolio returns.

    Parameters
    ----------
    month_returns : pd.DataFrame
        Daily returns for one month (coins as columns)
    weights : pd.Series
        Portfolio weights (must align with month_returns columns)

    Returns
    -------
    pd.Series
        Daily portfolio returns for that month
    """
    # Align weights with available assets
    weights = weights.reindex(month_returns.columns).fillna(0)
    portfolio_returns = month_returns.dot(weights)
    return portfolio_returns
