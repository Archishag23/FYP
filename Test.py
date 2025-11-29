import pandas as pd
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

prices = pd.read_excel('SPX_sectors_data.xlsx',sheet_name='Prices',header=[0,1],index_col=0)

prices.dropna(how='all',inplace=True)
prices = prices.ffill().bfill()
print("Data loaded and cleaned.")

sectors = pd.read_excel('SPX_sectors_data.xlsx',sheet_name='Sectors',header=0,index_col=0)

print("Sectors data loaded.")

# volatility = prices.pct_change().rolling(window=21).std() * (252 ** 0.5)
# volatility.dropna(how='all',inplace=True)

# print("Volatility calculated.")

# rolling_cov = prices.pct_change().rolling(window=21).cov()
# rolling_cov.dropna(how='all',inplace=True)
# rolling_cov.fillna(0, inplace=True)

# print("Rolling covariance calculated.")

# covariance_matrices = {
#     date: rolling_cov.xs(date, level=0) for date in rolling_cov.index.get_level_values(0).unique()
# }

rolling_corr = prices.pct_change().rolling(window=21).corr()
rolling_corr.dropna(how='all',inplace=True) 

print("Rolling correlation calculated.")

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

plt.plot(lambda_2_list)
plt.title('Algebraic Connectivity (λ2) Over Time')
plt.xlabel('Time')
plt.ylabel('λ2')
plt.show()

plt.plot(spectral_gap_list)
plt.title('Spectral Gap Over Time')     
plt.xlabel('Time')
plt.ylabel('Spectral Gap')
plt.show()