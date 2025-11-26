import pandas as pd
import networkx as nx
import numpy as np

prices = pd.read_excel('SPX_sectors_data.xlsx',sheet_name='Prices',header=[0,1],index_col=0)

prices.dropna(how='all',inplace=True)
prices = prices.ffill().bfill()
sectors = pd.read_excel('SPX_sectors_data.xlsx',sheet_name='Sectors',header=0,index_col=0)

volatility = prices.pct_change().rolling(window=21).std() * (252 ** 0.5)
volatility.dropna(how='all',inplace=True)

rolling_cov = prices.pct_change().rolling(window=21).cov()
rolling_cov.dropna(how='all',inplace=True)
rolling_cov.fillna(0, inplace=True)
covariance_matrices = {
    date: rolling_cov.xs(date, level=0) for date in rolling_cov.index.get_level_values(0).unique()
}

rolling_corr = prices.pct_change().rolling(window=21).corr()
rolling_corr.dropna(how='all',inplace=True) 

lambda_2_list = []
spectral_gap_list = []  

for date in rolling_corr.index:
    print(f"Processing date: {date}")
    corr_time = rolling_corr.loc[date]
    W = corr_time.clip(lower=0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    Q = -L
    eigenvalues = np.linalg.eigvalsh(L)  
    lambda_2 = eigenvalues[1]
    spectral_gap = eigenvalues[1] - eigenvalues[0]
    spectral_radius = np.max(np.abs(eigenvalues))
    lambda_2_list.append((date, lambda_2))
    spectral_gap_list.append((date, spectral_gap))

