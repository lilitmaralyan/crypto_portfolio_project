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

