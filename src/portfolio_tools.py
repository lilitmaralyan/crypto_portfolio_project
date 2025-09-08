import pandas as pd

def compute_equal_weights(monthly_returns):
    """
    Compute equal weights for each month.
    
    Parameters
    ----------
    monthly_returns : dict
        {month_string: DataFrame of daily returns for that month}
        
    Returns
    -------
    dict
        {month_string: pd.Series of weights per coin}
    """
    weights = {}
    for month, df in monthly_returns.items():
        n_coins = df.shape[1]
        weights[month] = pd.Series(1/n_coins, index=df.columns)
    return weights

def compute_portfolio_returns(monthly_returns: dict, monthly_weights: dict) -> pd.Series:
    """
    Compute daily portfolio returns using monthly weights.
    
    Parameters
    ----------
    monthly_returns : dict
        {month_string: DataFrame of daily returns for that month}
    monthly_weights : dict
        {month_string: Series of weights per coin}
    
    Returns
    -------
    pd.Series
        Daily portfolio returns indexed by date
    """
    portfolio_returns = []

    for month, returns_df in monthly_returns.items():
        weights = monthly_weights[month]
        # Multiply each column by its weight and sum across coins
        daily_portfolio = returns_df.dot(weights)
        portfolio_returns.append(daily_portfolio)

    # Concatenate all months into one Series
    portfolio_returns = pd.concat(portfolio_returns).sort_index()
    return portfolio_returns

