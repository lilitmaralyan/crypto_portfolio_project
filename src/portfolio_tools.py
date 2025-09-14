import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def compute_momentum_weights(prev_month_returns: pd.DataFrame, top_quantile: float = 0.2) -> pd.Series:
    """
    Computes momentum-based weights: long top quantile of coins 
    ranked by previous month's total return.

    Parameters
    ----------
    prev_month_returns : pd.DataFrame
        Daily returns for the previous month.
    top_quantile : float
        Fraction of top coins to select (default = 0.2 = top 20%).

    Returns
    -------
    pd.Series
        Equal weights among top momentum coins (sums to 1).
    """
    # 1. Compute cumulative return over previous month
    momentum = (1 + prev_month_returns).prod() - 1

    # 2. Find cutoff for top performers
    cutoff = momentum.quantile(1 - top_quantile)

    # 3. Select top quantile coins
    winners = momentum[momentum >= cutoff].index

    # 4. Equal weight among winners
    weights = pd.Series(0.0, index=momentum.index, dtype=float)
    if len(winners) > 0:
        weights[winners] = 1.0 / len(winners)

    return weights

def compute_reversal_weights(prev_month_returns: pd.DataFrame, bottom_quantile: float = 0.2) -> pd.Series:
    """
    Computes short-term reversal-based weights: long bottom quantile 
    of coins ranked by previous month's total return.

    Parameters
    ----------
    prev_month_returns : pd.DataFrame
        Daily returns for the previous month.
    bottom_quantile : float
        Fraction of bottom coins to select (default = 0.2 = bottom 20%).

    Returns
    -------
    pd.Series
        Equal weights among bottom momentum coins (sums to 1).
    """
    # 1. Compute cumulative return over previous month
    momentum = (1 + prev_month_returns).prod() - 1

    # 2. Find cutoff for worst performers
    cutoff = momentum.quantile(bottom_quantile)

    # 3. Select bottom quantile coins
    losers = momentum[momentum <= cutoff].index

    # 4. Equal weight among losers
    weights = pd.Series(0.0, index=momentum.index, dtype=float)
    if len(losers) > 0:
        weights[losers] = 1.0 / len(losers)

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

def evaluate_portfolio(portfolio_returns: pd.Series, freq: int = 252) -> dict:
    """
    Evaluates performance metrics for a portfolio in Paleologo style.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns (index = datetime).
    freq : int
        Trading frequency for annualization (252 = daily, 12 = monthly).

    Returns
    -------
    dict
        Dictionary of key performance metrics.
    """
    # Growth of 1 unit
    cumulative_return = (1 + portfolio_returns).prod() - 1

    # Annualized mean return
    annualized_return = (1 + portfolio_returns.mean()) ** freq - 1

    # Annualized volatility
    annualized_vol = portfolio_returns.std() * (freq ** 0.5)

    # Sharpe ratio
    sharpe = (annualized_return) / annualized_vol if annualized_vol > 0 else 0

    # Drawdowns
    wealth = (1 + portfolio_returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = annualized_return / abs(max_dd) if max_dd < 0 else None

    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar
    }

def plot_performance(portfolio_returns, title="Portfolio Performance"):
    wealth = (1 + portfolio_returns).cumprod()

    fig, ax = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    # Cumulative wealth
    ax[0].plot(wealth, label="Cumulative Return")
    ax[0].set_ylabel("Growth of $1")
    ax[0].legend()

    # Drawdown
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    ax[1].plot(drawdown, color="red", label="Drawdown")
    ax[1].set_ylabel("Drawdown")
    ax[1].legend()

    plt.suptitle(title)
    plt.show()
