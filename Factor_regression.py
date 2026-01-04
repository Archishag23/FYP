import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
#import cvxpy as cp
from scipy.optimize import minimize

# use different factors to model returns - use regression
# use realised vol, price-vol feedback rate, momentum, GARCH vol, bubbles_test as factors
# use PCA to work out which factors explain most variance in returns
# assume X = [1 RV GARCH_vol price-vol-feedback-ratio momentum bubbles_test]
# y = returns of SPX index

print("Loading SPX sector data...")
prices = pd.read_excel('SPX_data.xlsx', header=[0,1], index_col=0) # data for SPX as a whole
prices = prices[('PX_LAST')]
prices.dropna(how='all', inplace=True)
prices = prices.ffill().bfill()
prices = prices.loc['2007-01-01':]      

returns = prices.pct_change().dropna()

realised_volatility = returns.rolling(window=21).std() * (252 ** 0.5)
realised_volatility.dropna(how='all', inplace=True)

GARCH_variances = pd.read_csv("GARCH_variances.csv", index_col=0, parse_dates=True)
GARCH_volatility = np.sqrt(GARCH_variances)

Y = returns.shift(-1)

X = pd.concat([
    realised_volatility,
    GARCH_volatility
], axis=1)

X = sm.add_constant(X)
X.dropna(inplace=True) # X from 2007-02-02 to 2025-11-24
Y.dropna(inplace=True)
Y = Y.loc['2007-02-02':'2025-11-24'] # Y from 2007-02-02 to 2025-11-24
X = X.loc[:'2025-11-21']

model = sm.OLS(Y, X)
results = model.fit(cov_type="HAC", cov_kwds={"maxlags":5})
print(results.summary())
print("Coefficients:")
print(results.params)

