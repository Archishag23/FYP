import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

print("Loading SPX sector data...")
prices = pd.read_excel('SPX_data.xlsx', header=[0,1], index_col=0)
prices.dropna(how='all', inplace=True)
prices = prices.squeeze()
prices = prices.ffill().bfill()
prices = prices.loc['2019-01-01':'2020-12-31']

volatility = prices.pct_change().rolling(window=21).std() * (252 ** 0.5)
volatility.dropna(how='all', inplace=True)
print("Volatility calculated.")

# market_caps = pd.read_excel('SPX_data.xlsx', header=[0,1], index_col=0, sheet_name='Market_Caps')
# market_caps.dropna(how='all', inplace=True)
# market_caps = market_caps.squeeze()
# market_caps = market_caps.ffill().bfill()
# print("Market capitalisations loaded and cleaned.")

n = 1000 # Number of simulations
var_95_list = []    

returns = prices.pct_change().dropna()
returns = returns.sort_values()
var_value = 0.05 * len(returns)
var_95_historical = returns.iloc[int(var_value)]
print(f"Historical VaR 95%: {var_95_historical}")

loss = returns.iloc[:int(var_value)]
ES = np.average(loss)
print(f"Historical ES 95%: {ES}")

# for date in volatility.index:
#     vol_time = volatility.xs(date, level=0)
#     cap_time = market_caps.xs(date, level=0)
#     weights = cap_time / cap_time.sum()
#     cov_matrix = vol_time.to_frame().dot(vol_time.to_frame().T)
#     mean_returns = np.zeros(len(vol_time))
#     simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n)
#     portfolio_returns = simulated_returns.dot(weights)
#     var_95 = np.percentile(portfolio_returns, 5)
#     var_95_list.append((date, var_95))
#     print(f"Date: {date}, VaR 95%: {var_95}")
# # Convert results to DataFrame for easier handling
# var_95_df = pd.DataFrame(var_95_list, columns=['Date', 'VaR_95'])
# # Plot VaR over time
# plt.plot(var_95_df['Date'], var_95_df['VaR_95'])
# plt.title('Value at Risk (VaR) 95% Over Time')
# plt.xlabel('Time')  
# plt.ylabel('VaR 95%')
# plt.show()
