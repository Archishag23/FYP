import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- 1. Load Data ---
print("Loading data...")

# Helper to load and fix dates
def load_and_fix_index(filename):
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index, format='%m/%d/%Y') # Fix date format
    df.dropna(how='all', inplace=True)
    df = df.ffill().bfill()
    return df


# Load Constituent Factors (Tables where Cols = Tickers, Rows = Dates)
Market_caps = load_and_fix_index('Data/SPX_Constituents_market_cap_2006_2025(in).csv')
train_market_caps = Market_caps.loc['2007-01-01':'2015-12-31']
test_market_caps = Market_caps.loc['2016-01-01':]

PE_ratios = load_and_fix_index('Data/SPX_Constituents_Calculated_PE_2006_2025(in).csv')
train_pe_ratios = PE_ratios.loc['2007-01-01':'2015-12-31']
test_pe_ratios = PE_ratios.loc['2016-01-01':]

Implied_vol = load_and_fix_index('Data/SPX_Constituents_Implied_vol_2006_2025(in).csv')
train_implied_vol = Implied_vol.loc['2007-01-01':'2015-12-31']
test_implied_vol = Implied_vol.loc['2016-01-01':]

Short_interest = load_and_fix_index('Data/SPX_Constituents_Short_Interest_Pct_2006_2025(in).csv')
train_short_interest = Short_interest.loc['2007-01-01':'2015-12-31']
test_short_interest = Short_interest.loc['2016-01-01':]

Beta = load_and_fix_index('Data/SPX_Constituents_Beta_2006_2025(in).csv')
train_beta = Beta.loc['2007-01-01':'2015-12-31']
test_beta = Beta.loc['2016-01-01':]

Operating_margin = load_and_fix_index('Data/SPX_Constituents_Op_Margin_2006_2025(in).csv')   
train_operating_margin = Operating_margin.loc['2007-01-01':'2015-12-31']
test_operating_margin = Operating_margin.loc['2016-01-01':]

Return_on_equity = load_and_fix_index('Data/SPX_Constituents_Ret_On_Equity_2006_2025(in).csv')
train_return_on_equity = Return_on_equity.loc['2007-01-01':'2015-12-31']
test_return_on_equity = Return_on_equity.loc['2016-01-01':]

RSI_momentum = load_and_fix_index('Data/SPX_Constituents_RSI_momentum_2006_2025(in).csv')
train_rsi_momentum = RSI_momentum.loc['2007-01-01':'2015-12-31']
test_rsi_momentum = RSI_momentum.loc['2016-01-01':]

Turnover = load_and_fix_index('Data/SPX_Constituents_Turnover_30D_2006_2025(in).csv')
train_turnover = Turnover.loc['2007-01-01':'2015-12-31']
test_turnover = Turnover.loc['2016-01-01':]


# Load Prices (Make sure this contains MANY columns, not just one)
prices = pd.read_excel('SPX_sectors_data.xlsx', header=[0,1], index_col=0)
prices.dropna(how='all', inplace=True)
prices = prices.ffill().bfill()
prices.columns = prices.columns.droplevel(1)
train_prices = prices.loc['2007-01-01':'2015-12-31']
test_prices = prices.loc['2016-01-01':]

# Calculate Returns for ALL stocks
train_returns = train_prices.pct_change()
test_returns = test_prices.pct_change()

# Calculate Realised Volatility for ALL stocks
train_realised_volatility = train_returns.rolling(window=21).std() * (252 ** 0.5)
test_realised_volatility = test_returns.rolling(window=21).std() * (252 ** 0.5)

# --- 2. Prepare for the Loop ---

# Find the list of tickers that exist in dataframes
# (Intersection of columns)
valid_tickers = train_prices.columns \
    .intersection(train_pe_ratios.columns) \
    .intersection(train_beta.columns) \
    .intersection(train_short_interest.columns) \
    .intersection(train_implied_vol.columns) \
    .intersection(train_realised_volatility.columns) \
    .intersection(train_return_on_equity.columns) \
    .intersection(train_rsi_momentum.columns) \
    .intersection(train_turnover.columns) \
    .intersection(train_market_caps.columns) \
    .intersection(train_operating_margin.columns)

print(f"Found {len(valid_tickers)} common tickers to analyze.")

# DataFrame to store your results (e.g., the Beta coefficient for each stock)
factor_sensitivities = pd.DataFrame(index=valid_tickers, columns=['Realized_Vol', 'PE', 'Imp_Vol', 'Short_Int', 'Beta_Factor', 'Market_Cap', 'Op_Margin', 'Ret_On_Equity', 'RSI_Momentum', 'Turnover'])

# --- 3. The Regression Loop ---
print("Starting regression loop...")

for ticker in valid_tickers:
    try:
        # A. Build the Target (Y) for THIS stock
        # Shift -1 because we predict tomorrow's return using today's factors
        Y = train_returns[ticker].shift(-1)
        
        # B. Build the Features (X) for THIS stock
        # We pick the specific column for this ticker from each factor DF
        X = pd.DataFrame({
            'Realized_Vol': train_realised_volatility[ticker],
            'PE': train_pe_ratios[ticker],
            'Imp_Vol': train_implied_vol[ticker],
            'Short_Int': train_short_interest[ticker],
            'Beta_Factor': train_beta[ticker],
            'Market_Cap': train_market_caps[ticker],
            'Op_Margin': train_operating_margin[ticker],
            'Ret_On_Equity': train_return_on_equity[ticker],
            'RSI_Momentum': train_rsi_momentum[ticker],
            'Turnover': train_turnover[ticker]
        })
        
        # C. Align and Clean
        #X = sm.add_constant(X)
        
        # Combine temporary to drop NaNs together
        data = pd.concat([Y, X], axis=1).dropna()
        
        if len(data) < 50: # Skip stocks with not enough history
            continue
            
        # Split back out
        Y_clean = data.iloc[:, 0]
        X_clean = data.iloc[:, 1:]
        
        # D. Run Regression
        model = sm.OLS(Y_clean, X_clean)
        results = model.fit() # HAC is slow in a loop, maybe skip for speed unless necessary
        
        # E. Store Results
        # results.params contains the coefficients (alphas/betas)
        factor_sensitivities.loc[ticker] = results.params
        
    except Exception as e:
        print(f"Could not fit {ticker}: {e}")

# --- 4. View Results ---
print("\nRegression Complete. Here are the factor sensitivities per stock:")
print(factor_sensitivities.head())

factor_sensitivities.to_csv("Factor_Betas.csv")

#testing betas
print("\nTesting factor sensitivities on test data...")
errors = pd.DataFrame(index=valid_tickers, columns=['MSE'])
for ticker in valid_tickers:
    try:
        # A. Build the Target (Y) for THIS stock
        Y_test = test_returns[ticker].shift(-1)
        
        # B. Build the Features (X) for THIS stock
        X_test = pd.DataFrame({
            'Realized_Vol': test_realised_volatility[ticker],
            'PE': test_pe_ratios[ticker],
            'Imp_Vol': test_implied_vol[ticker],
            'Short_Int': test_short_interest[ticker],
            'Beta_Factor': test_beta[ticker],
            'Market_Cap': test_market_caps[ticker],
            'Op_Margin': test_operating_margin[ticker],
            'Ret_On_Equity': test_return_on_equity[ticker],
            'RSI_Momentum': test_rsi_momentum[ticker],
            'Turnover': test_turnover[ticker]
        })
        
        # C. Align and Clean
        #X_test = sm.add_constant(X_test)
        
        # Combine temporary to drop NaNs together
        data_test = pd.concat([Y_test, X_test], axis=1).dropna()
        
        if len(data_test) < 50: # Skip stocks with not enough history
            continue
            
        # Split back out
        Y_test_clean = data_test.iloc[:, 0]
        X_test_clean = data_test.iloc[:, 1:]
        
        # D. Predict using stored betas
        betas = factor_sensitivities.loc[ticker].values.astype(float)
        predictions = np.dot(X_test_clean, betas)
        
        # E. Evaluate Predictions (e.g., MSE)
        mse = np.mean((Y_test_clean - predictions) ** 2)
        print(f"{ticker} - Test MSE: {mse:.6f}")
        errors.loc[ticker] = mse
        
    except Exception as e:
        print(f"Could not test {ticker}: {e}")

errors.to_csv("Test_MSEs.csv")

plt.plot(errors.index, errors['MSE'])
plt.xticks(rotation=90)
plt.ylabel("Mean Squared Error")
plt.title("Test MSE per Stock")
plt.tight_layout()
plt.show()

