import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings('ignore')

#CoVaR based on Adrian & Brunnermeier (2011)
# VaR_i^q means value at Risk of institution i at quantile q
# CoVaR_{j|i}^q is  VaR of system j conditional on institution i being at its VaR
# ΔCoVaR_i = Contribution to systemic risk = CoVaR_{sys|i=VaR} - CoVaR_{sys|i=median}

class CoVaRModel:
    
    def __init__(self, quantile=0.05):
        
        
        #quantile: Risk quantile (0.05 for 5% VaR, 0.01 for 1% VaR)
        self.quantile = quantile
        self.var_models = {}
        self.covar_models = {}
        self.results = {}
        
    def calculate_var(self, returns, state_variables=None):
        """
        Calculate VaR using quantile regression
        
        Returns ~ α + β'X_{t-1} + ε
        
        Args:
            returns: Series of asset returns
            state_variables: DataFrame of lagged state variables (optional)
        """
        data = pd.DataFrame({'returns': returns}).dropna()
        
        if state_variables is not None:
            # Merge with state variables
            data = data.join(state_variables, how='inner')
            X = data[state_variables.columns]
        else:
            # Just use constant
            X = pd.DataFrame({'const': 1}, index=data.index)
        
        y = data['returns']
        
        # Fit quantile regression
        model = QuantReg(y, X)
        fitted = model.fit(q=self.quantile)
        
        # Predicted VaR (conditional quantile)
        var_predicted = fitted.predict(X)
        
        return {
            'model': fitted,
            'var_predicted': var_predicted,
            'var_unconditional': returns.quantile(self.quantile)
        }
    
    def calculate_covar(self, system_returns, institution_returns, state_variables=None):
        """
        Calculate CoVaR: VaR of system conditional on institution being at its VaR
        
        System_returns ~ α + β * Institution_returns + γ'X_{t-1} + ε
        
        Args:
            system_returns: Returns of the financial system (or sector aggregate)
            institution_returns: Returns of individual institution
            state_variables: DataFrame of lagged state variables (optional)
        """
        # Step 1: Calculate institution's VaR
        var_result = self.calculate_var(institution_returns, state_variables)
        var_q = var_result['var_unconditional']
        var_median = institution_returns.median()
        
        # Step 2: Quantile regression of system on institution
        data = pd.DataFrame({
            'system': system_returns,
            'institution': institution_returns
        }).dropna()
        
        if state_variables is not None:
            data = data.join(state_variables, how='inner')
            X_cols = ['institution'] + list(state_variables.columns)
            X = data[X_cols]
        else:
            X = data[['institution']]
        
        y = data['system']
        
        # Add constant
        X = X.copy()
        X.insert(0, 'const', 1)
        
        # Fit quantile regression
        model = QuantReg(y, X)
        fitted = model.fit(q=self.quantile)
        
        # Step 3: Calculate CoVaR by evaluating at institution's VaR
        # CoVaR^q_{sys|i=VaR^q_i} = α^q + β^q * VaR^q_i + γ^q' X
        
        # Create prediction data at institution VaR level
        X_var = X.copy()
        X_var['institution'] = var_q
        covar_at_var = fitted.predict(X_var).mean()
        
        # Create prediction data at institution median level
        X_median = X.copy()
        X_median['institution'] = var_median
        covar_at_median = fitted.predict(X_median).mean()
        
        # Step 4: Calculate ΔCoVaR
        delta_covar = covar_at_var - covar_at_median
        
        return {
            'model': fitted,
            'covar_at_var': covar_at_var,
            'covar_at_median': covar_at_median,
            'delta_covar': delta_covar,
            'institution_var': var_q,
            'beta': fitted.params.get('institution', np.nan)
        }
    
    def fit_panel(self, returns_df, state_variables=None):
        """
        Fit CoVaR for all institutions in a panel
        
        Args:
            returns_df: DataFrame where columns are institutions, rows are dates
            state_variables: DataFrame of state variables (optional)
        """
        # Calculate system returns (equal-weighted average)
        system_returns = returns_df.mean(axis=1)
        
        results = {}
        
        for col in returns_df.columns:
            print(f"Calculating CoVaR for {col}...")
            
            institution_returns = returns_df[col].dropna()
            
            try:
                # Calculate VaR
                var_result = self.calculate_var(institution_returns, state_variables)
                
                # Calculate CoVaR
                covar_result = self.calculate_covar(
                    system_returns, 
                    institution_returns, 
                    state_variables
                )
                
                results[col] = {
                    'var': var_result['var_unconditional'],
                    'delta_covar': covar_result['delta_covar'],
                    'covar_at_var': covar_result['covar_at_var'],
                    'covar_at_median': covar_result['covar_at_median'],
                    'beta': covar_result['beta']
                }
                
            except Exception as e:
                print(f"  Error: {e}")
                results[col] = {
                    'var': np.nan,
                    'delta_covar': np.nan,
                    'covar_at_var': np.nan,
                    'covar_at_median': np.nan,
                    'beta': np.nan
                }
        
        self.results = results
        return pd.DataFrame(results).T



print("Loading SPX sector data...")
prices = pd.read_excel('SPX_sectors_data.xlsx', sheet_name='Prices', 
                       header=[0,1], index_col=0)
prices.dropna(how='all', inplace=True)
prices = prices.ffill().bfill()

sectors = pd.read_excel('SPX_sectors_data.xlsx', sheet_name='Sectors', 
                        header=0, index_col=0)

# Flatten multi-index columns if needed
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(0)

print(f"Loaded {len(prices.columns)} stocks from {prices.index[0]} to {prices.index[-1]}")

# Calculate returns
returns = prices.pct_change().dropna()

# Aggregate to sector level (equal-weighted sector returns)
print("\nAggregating to sector level...")
sector_returns = pd.DataFrame()
for sector in sectors['Sector'].unique():
    sector_stocks = sectors[sectors['Sector'] == sector].index
    # Find stocks that exist in both sectors and returns
    valid_stocks = [s for s in sector_stocks if s in returns.columns]
    if len(valid_stocks) > 0:
        sector_returns[sector] = returns[valid_stocks].mean(axis=1)

print(f"Sector returns calculated for {len(sector_returns.columns)} sectors")
print(f"Sectors: {list(sector_returns.columns)}")


print("\nCreating state variables...")

# Use sector returns to create state variables (much more memory efficient)
state_vars = pd.DataFrame(index=sector_returns.index)

# 1. Market volatility (rolling 21-day std)
market_returns = sector_returns.mean(axis=1)
state_vars['volatility'] = market_returns.rolling(21).std() * np.sqrt(252)

# 2. Market return (lagged)
state_vars['market_return'] = market_returns.shift(1)

# 3. Average correlation across sectors (memory efficient version)
print("  Computing rolling correlation...")
window = 63
avg_corr_list = []

for i in range(len(sector_returns)):
    if i < window:
        avg_corr_list.append(np.nan)
    else:
        # Get window of data
        window_data = sector_returns.iloc[i-window:i]
        # Calculate correlation matrix
        corr_matrix = window_data.corr()
        # Get upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_tri_values = corr_matrix.where(mask).stack()
        # Average correlation
        avg_corr = upper_tri_values.mean()
        avg_corr_list.append(avg_corr)

state_vars['avg_correlation'] = avg_corr_list
state_vars['avg_correlation'] = state_vars['avg_correlation'].shift(1)

# 4. Term spread proxy (21-day vs 63-day moving average spread)
ma_21 = market_returns.rolling(21).mean()
ma_63 = market_returns.rolling(63).mean()
state_vars['term_spread'] = (ma_63 - ma_21).shift(1)

# 5. Credit spread proxy (rolling max drawdown)
cumulative = (1 + market_returns).cumprod()
rolling_max = cumulative.rolling(window=252, min_periods=1).max()
drawdown = (cumulative - rolling_max) / rolling_max
state_vars['credit_spread'] = drawdown.rolling(63).mean().shift(1)

# Clean state variables
state_vars = state_vars.dropna()

print(f"State variables created: {list(state_vars.columns)}")
print(f"Data range: {state_vars.index[0]} to {state_vars.index[-1]}")


print("\n" + "="*70)
print("CALCULATING CoVaR FOR SPX SECTORS")
print("="*70)

# Initialize CoVaR model (5% quantile)
covar_model_5pct = CoVaRModel(quantile=0.05)
covar_model_1pct = CoVaRModel(quantile=0.01)

# Calculate for each sector
print("\n--- 5% CoVaR Results ---")
results_5pct = covar_model_5pct.fit_panel(sector_returns, state_vars)
print("\n" + results_5pct.sort_values('delta_covar').to_string())

print("\n--- 1% CoVaR Results ---")
results_1pct = covar_model_1pct.fit_panel(sector_returns, state_vars)
print("\n" + results_1pct.sort_values('delta_covar').to_string())


print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. VaR vs ΔCoVaR scatter plot (replicating Figure 1 from paper)
ax = axes[0, 0]
ax.scatter(results_5pct['var'], results_5pct['delta_covar'], 
           s=100, alpha=0.6, c='steelblue')
for sector in results_5pct.index:
    ax.annotate(sector, 
                (results_5pct.loc[sector, 'var'], 
                 results_5pct.loc[sector, 'delta_covar']),
                fontsize=8)
ax.set_xlabel('Sector VaR (5%)', fontsize=12)
ax.set_ylabel('Sector ΔCoVaR (5%)', fontsize=12)
ax.set_title('VaR vs ΔCoVaR: Weak Cross-Sectional Link', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)

# 2. ΔCoVaR ranking (bar chart)
ax = axes[0, 1]
sorted_results = results_5pct.sort_values('delta_covar')
colors = ['red' if x < 0 else 'green' for x in sorted_results['delta_covar']]
ax.barh(sorted_results.index, sorted_results['delta_covar'], color=colors, alpha=0.7)
ax.set_xlabel('ΔCoVaR (5%)', fontsize=12)
ax.set_title('Systemic Risk Contribution by Sector', fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# 3. Comparison of 1% vs 5% ΔCoVaR
ax = axes[1, 0]
comparison = pd.DataFrame({
    '5% ΔCoVaR': results_5pct['delta_covar'],
    '1% ΔCoVaR': results_1pct['delta_covar']
})
comparison.plot(kind='bar', ax=ax, width=0.8)
ax.set_ylabel('ΔCoVaR', fontsize=12)
ax.set_title('Systemic Risk at Different Quantiles', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fancybox=True)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 4. Beta coefficients (sensitivity to system)
ax = axes[1, 1]
sorted_beta = results_5pct.sort_values('beta')
ax.barh(sorted_beta.index, sorted_beta['beta'], alpha=0.7, color='coral')
ax.set_xlabel('Beta (System Sensitivity)', fontsize=12)
ax.set_title('Sector Sensitivity to System Risk', fontsize=13, fontweight='bold')
ax.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Market Beta = 1')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('covar_analysis.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'covar_analysis.png'")
plt.show()



print("\n" + "="*70)
print("TIME-VARYING CoVaR ANALYSIS")
print("="*70)

def calculate_time_varying_covar(sector_name, sector_returns, system_returns, 
                                  state_vars, quantile=0.05):
    """Calculate time-varying ΔCoVaR over rolling windows"""
    
    window_size = 252 * 2  # 2 years
    results_list = []
    
    for i in range(window_size, len(sector_returns), 21):  # Every 21 days to speed up
        window_sector = sector_returns.iloc[i-window_size:i]
        window_system = system_returns.iloc[i-window_size:i]
        window_state = state_vars.iloc[i-window_size:i]
        
        # Align data
        data = pd.DataFrame({
            'system': window_system,
            'sector': window_sector
        }).join(window_state, how='inner').dropna()
        
        if len(data) < 100:  # Need enough data
            continue
        
        # Calculate sector VaR
        sector_var = data['sector'].quantile(quantile)
        sector_median = data['sector'].median()
        
        # Quantile regression
        X = data[['sector'] + list(window_state.columns)].copy()
        X.insert(0, 'const', 1)
        y = data['system']
        
        try:
            model = QuantReg(y, X).fit(q=quantile, max_iter=1000)
            
            # Predict CoVaR at sector VaR and median
            X_var = X.iloc[-1:].copy()
            X_var['sector'] = sector_var
            covar_var = model.predict(X_var).iloc[0]
            
            X_median = X.iloc[-1:].copy()
            X_median['sector'] = sector_median
            covar_median = model.predict(X_median).iloc[0]
            
            delta_covar = covar_var - covar_median
            
            results_list.append({
                'date': sector_returns.index[i],
                'delta_covar': delta_covar,
                'var': sector_var,
                'beta': model.params.get('sector', np.nan)
            })
        except:
            continue
    
    return pd.DataFrame(results_list).set_index('date')

# Calculate time-varying CoVaR for top 3 systemically important sectors
system_returns = sector_returns.mean(axis=1)
top_sectors = results_5pct.nsmallest(3, 'delta_covar').index

fig, axes = plt.subplots(len(top_sectors), 1, figsize=(14, 4*len(top_sectors)))
if len(top_sectors) == 1:
    axes = [axes]

for idx, sector in enumerate(top_sectors):
    print(f"\nCalculating time-varying CoVaR for {sector}...")
    tv_covar = calculate_time_varying_covar(
        sector, 
        sector_returns[sector], 
        system_returns, 
        state_vars,
        quantile=0.05
    )
    
    if len(tv_covar) == 0:
        print(f"  No results for {sector}")
        continue
    
    ax = axes[idx]
    ax.plot(tv_covar.index, tv_covar['delta_covar'], 
            linewidth=2, color='darkred', label='ΔCoVaR')
    ax.fill_between(tv_covar.index, tv_covar['delta_covar'], 0, 
                     alpha=0.3, color='red')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('ΔCoVaR', fontsize=11)
    ax.set_title(f'{sector} - Time-Varying Systemic Risk Contribution', 
                 fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    
    # crisis periods
    crisis_periods = [
        ('2008-01-01', '2009-06-30'),  # Financial Crisis
        ('2020-02-01', '2020-06-30'),  # COVID
    ]
    for start, end in crisis_periods:
        try:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), 
                       alpha=0.2, color='gray')
        except:
            pass

plt.tight_layout()
plt.savefig('time_varying_covar.png', dpi=150, bbox_inches='tight')
print("\nSaved time-varying CoVaR to 'time_varying_covar.png'")
plt.show()

print("\n" + "="*70)
print("SUMMARY: CoVaR SYSTEMIC RISK ANALYSIS")
print("="*70)

print("\nMost Systemically Important Sectors (5% ΔCoVaR):")
top_systemic = results_5pct.nsmallest(5, 'delta_covar')
for i, (sector, row) in enumerate(top_systemic.iterrows(), 1):
    print(f"{i}. {sector:30s} ΔCoVaR: {row['delta_covar']:.4f}")

print("\nLeast Systemically Important Sectors (5% ΔCoVaR):")
bottom_systemic = results_5pct.nlargest(3, 'delta_covar')
for i, (sector, row) in enumerate(bottom_systemic.iterrows(), 1):
    print(f"{i}. {sector:30s} ΔCoVaR: {row['delta_covar']:.4f}")

print("\nKey Insights:")
print("- ΔCoVaR measures how much system risk increases when a sector is in distress")
print("- More negative ΔCoVaR = higher contribution to systemic risk")
print("- Unlike VaR, ΔCoVaR captures risk spillovers and interconnectedness")
print("- Results can inform macroprudential policy and capital requirements")

# Save results to CSV
results_5pct.to_csv('covar_results_5pct.csv')
results_1pct.to_csv('covar_results_1pct.csv')
print("\nResults saved to 'covar_results_5pct.csv' and 'covar_results_1pct.csv'")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
