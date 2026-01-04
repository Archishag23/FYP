import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
#import cvxpy as cp
from scipy.optimize import minimize

print("Loading SPX sector data...")
prices = pd.read_excel('SPX_data.xlsx', header=[0,1], index_col=0)
prices = prices[('PX_LAST')]
prices.dropna(how='all', inplace=True)
prices = prices.ffill().bfill()
prices = prices.loc['2007-01-01':]      
prices = prices.iloc[:, 0]

#sigma_n squared = w + alpha*mu_(n-1)^2 + beta*sigma_(n-1)^2

variances = [] # length n-2

returns = 100 * prices.pct_change().dropna()
dates = returns.index
mu_squared = (returns.to_numpy() ** 2)
mu_squared = pd.Series(mu_squared, index=dates)
mu_squared.to_csv("Realised_variance.csv")
# w, alpha, beta = 0.000001, 0.1, 0.85
# variances[0] = mu_squared[1]
# for i in range(1, len(mu_squared)-2):
#     variances[i] = w + alpha * mu_squared[i-1] + beta * variances[i-1]

# constraint = [w + alpha + beta == 1]
# objective = cp.maximise(variances.sum())
# problem = cp.Problem(objective, constraint)
# problem.solve()
# print(f"Optimized parameters: w={w}, alpha={alpha}, beta={beta}")

def nll(p):
    w, a, b = p
    if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
        return 1e50

    v = np.empty(len(mu_squared), dtype=float)
    v[0] = mu_squared.mean()
    ms = mu_squared.to_numpy()
    for t in range(1, len(ms)):
        v[t] = w + a * ms[t-1] + b * v[t-1]
        if v[t] <= 1e-12:
            return 1e50

    return 0.5 * np.sum(np.log(v) + mu_squared / v)
res = minimize(
    nll,
    x0=(1e-6, 0.05, 0.90),
    method="SLSQP",
    bounds=[(1e-12, None), (0, 1), (0, 1)],
    constraints=[{"type": "ineq", "fun": lambda p: 0.999 - (p[1] + p[2])}],
)
w_opt, a_opt, b_opt = res.x
print(f"Optimized parameters: w={w_opt}, alpha={a_opt}, beta={b_opt}")
print(w_opt + a_opt + b_opt)
print(f"Long term vol rate (daily): {np.sqrt(w_opt/(1 - a_opt - b_opt))}")

variances = pd.Series(index=dates, dtype=float)

variances.iloc[0] = mu_squared.mean()

for t in range(1, len(mu_squared)):
    variances.iloc[t] = (
        w_opt
        + a_opt * mu_squared[t-1]
        + b_opt * variances.iloc[t-1]
    )
variances.to_csv("GARCH_variances.csv")

plt.figure(figsize=(10, 5))
plt.plot(np.sqrt(mu_squared), label='Realised Vol', alpha=0.6)
plt.plot(np.sqrt(variances), label='GARCH(1,1) Vol', linewidth=2)

plt.title('GARCH(1,1) Vol vs Realised Vol')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.tight_layout()
plt.savefig('volatility.png', dpi=150, bbox_inches='tight')
plt.show()

