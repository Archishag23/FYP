import pandas as pd
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

prices = pd.read_excel('SPX_sectors_data.xlsx',sheet_name='Prices',header=[0,1],index_col=0)

prices.dropna(how='all',inplace=True)
prices = prices.ffill().bfill()
prices = prices.loc['2019-01-01':'2020-12-31']
print("Data loaded and cleaned.")

sectors = pd.read_excel('SPX_sectors_data.xlsx',sheet_name='Sectors',header=0,index_col=0)

print("Sectors data loaded.")

# memory issues with covariance, corr matrices for individual dates for stocks
rolling_corr = prices.pct_change().rolling(window=50).corr()
rolling_corr.dropna(how='all',inplace=True) 

print("Rolling correlation calculated.")

lambda_2_list = []
spectral_gap_list = []  

dates = rolling_corr.index.get_level_values(0).unique()

for date in dates:
    print(f"Processing date: {date}")
    corr_time = rolling_corr.xs(date, level=0)
    common = corr_time.index.intersection(corr_time.columns)
    corr_time = corr_time.loc[common, common]
    corr_time = corr_time.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    W = corr_time.clip(lower=0).to_numpy()
    D = np.diag(W.sum(axis=1))
    L = D - W
    Q = -L
    eigenvalues = np.linalg.eigvalsh(L)  
    lambda_2 = eigenvalues[1]
    spectral_gap = eigenvalues[1] - eigenvalues[0]
    spectral_radius = np.max(np.abs(eigenvalues))
    lambda_2_list.append(lambda_2)
    spectral_gap_list.append(spectral_gap)
    print(f"Date: {date}, λ2: {lambda_2}")

plt.plot(dates, lambda_2_list)
plt.title('Algebraic Connectivity (λ2) Over Time')
plt.xlabel('Time')
plt.ylabel('λ2')
plt.show()

