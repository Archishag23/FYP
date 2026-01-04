import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


print("Loading SPX sector data...")
prices = pd.read_excel('SPX_data.xlsx', header=[0,1], index_col=0)
prices.dropna(how='all', inplace=True)
prices = prices.squeeze()
prices = prices.ffill().bfill()
#choose prices for dates from 1/1/2022 to latest date:
prices = prices.loc['2022-01-01':]
print("Data loaded.")
#inputting the parameters
r0 = int(len(prices)*0.1)
#print(r0)
#specify lags for the ADF)test
adf_lags = 3
#critical value of the right-tailed ADF-test (95%) from Phillips et al. (2015)
crit = 1.49
#transforming data
log_prices = np.array(np.log(prices))
delta_log_prices = log_prices[1:] - log_prices[:-1]
n = len(delta_log_prices)
BSADF = np.array([])
#calculating ADF stats
for r2 in range(r0,n):
    ADFS = np.array([])
    for r1 in range(0,r2-r0+1):
        X0 = log_prices[r1:r2+1]
        X = pd.DataFrame()
        X[0] = X0
        for j in range(1,adf_lags+1):
            X[j] = np.append(np.zeros(j),delta_log_prices[r1:r2+1-j])
        X = np.array(X)
        Y = delta_log_prices[r1:r2+1]
        reg = sm.OLS(Y,sm.add_constant(X))
        res = reg.fit()
        ADFS = np.append(ADFS, res.params[1]/res.bse[1])
    BSADF = np.append(BSADF, max(ADFS))
    print(f"Completed {r2-r0+1} of {n - r0} steps.")
#visualising the results
plt.rc('xtick',labelsize = 8)
plt.plot(prices.index[r0+1:],BSADF)
plt.plot(prices.index[r0+1:],np.ones(len(BSADF))*crit)
plt.show() 
#printing dates when bubbles were detected
print(prices.index[r0+1:][BSADF > crit])

