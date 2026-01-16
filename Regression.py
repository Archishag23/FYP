import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# --- Configuration ---
DATA_FILES = {
    'Market_caps': 'Data/SPX_Constituents_market_cap_2006_2025(in).csv',
    'PE_ratios': 'Data/SPX_Constituents_Calculated_PE_2006_2025(in).csv',
    'Implied_vol': 'Data/SPX_Constituents_Implied_vol_2006_2025(in).csv',
    'Short_interest': 'Data/SPX_Constituents_Short_Interest_Pct_2006_2025(in).csv',
    'Beta': 'Data/SPX_Constituents_Beta_2006_2025(in).csv',
    'Operating_margin': 'Data/SPX_Constituents_Op_Margin_2006_2025(in).csv',
    'Return_on_equity': 'Data/SPX_Constituents_Ret_On_Equity_2006_2025(in).csv',
    'RSI_momentum': 'Data/SPX_Constituents_RSI_momentum_2006_2025(in).csv',
    'Turnover': 'Data/SPX_Constituents_Turnover_30D_2006_2025(in).csv'
}
PRICES_FILE = 'SPX_sectors_data.xlsx'

# --- 1. Data Loading & Preprocessing ---

def load_and_fix_index(filename):
    """Loads a CSV, parses dates, and fills missing values."""
    print(f"Loading {filename}...")
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index, format='%m/%d/%Y', errors='coerce')
    df.dropna(how='all', inplace=True)
    df = df.ffill().bfill()
    return df

def load_all_data():
    """Loads all factor files and price data."""
    factors = {}
    for name, path in DATA_FILES.items():
        factors[name] = load_and_fix_index(path)
    
    print(f"Loading {PRICES_FILE}...")
    prices = pd.read_excel(PRICES_FILE, header=[0, 1], index_col=0)
    prices.dropna(how='all', inplace=True)
    prices = prices.ffill().bfill()
    prices.columns = prices.columns.droplevel(1)
    
    return factors, prices

def calculate_derived_metrics(prices):
    """Calculates Returns and Realized Volatility."""
    returns = prices.pct_change()
    # Annualized Volatility (21-day rolling)
    realized_vol = returns.rolling(window=21).std() * (252 ** 0.5)
    return returns, realized_vol

def split_train_test(df, split_date='2016-01-01'):
    """Splits a DataFrame into train and test sets."""
    train = df.loc[:split_date]
    test = df.loc[split_date:]
    # Drop the last row of train/first of test overlap if necessary, 
    # but slicing by date usually handles this cleanly.
    return train, test

def get_valid_tickers(factors, prices):
    """Finds the intersection of tickers across all DataFrames."""
    valid_tickers = prices.columns
    for df in factors.values():
        valid_tickers = valid_tickers.intersection(df.columns)
    
    # Also intersect with calculated metrics if they aren't in 'factors' yet
    # (Assuming they will be aligned later, but good to check)
    return valid_tickers

# --- 2. Model Training ---

def build_feature_set(ticker, date_idx, factors, realized_vol):
    """Helper to construct the X matrix for a specific ticker."""
    # This assumes 'factors' is a dict of DataFrames
    data = {name: df.loc[date_idx, ticker] for name, df in factors.items()}
    data['Realized_Vol'] = realized_vol.loc[date_idx, ticker]
    return pd.DataFrame(data)

def train_model(valid_tickers, train_returns, train_factors_dict, train_realized_vol):
    """Runs OLS regression for each ticker to get factor betas."""
    print("Starting training loop...")
    factor_sensitivities = pd.DataFrame(index=valid_tickers, 
                                        columns=list(train_factors_dict.keys()) + ['Realized_Vol'])
    
    for ticker in valid_tickers:
        try:
            # A. Build Target (Y): Next day's return
            Y = train_returns[ticker].shift(-1)
            
            # B. Build Features (X)
            X = build_feature_set(ticker, train_returns.index, train_factors_dict, train_realized_vol)
            
            # C. Align and Clean
            data = pd.concat([Y, X], axis=1).dropna()
            
            if len(data) < 50:
                continue
                
            Y_clean = data.iloc[:, 0]
            X_clean = data.iloc[:, 1:]
            
            # D. Run Regression
            model = sm.OLS(Y_clean, X_clean) # Add sm.add_constant(X_clean) if you want Alpha
            results = model.fit()
            
            factor_sensitivities.loc[ticker] = results.params
            
        except Exception as e:
            # print(f"Error training {ticker}: {e}") # Optional: suppress noise
            pass
            
    print("Training complete.")
    return factor_sensitivities.dropna()

# --- 3. Model Evaluation (ROC-AUC) ---

def evaluate_model(valid_tickers, test_returns, test_factors_dict, test_realized_vol, factor_sensitivities):
    """Tests the model and calculates MSE and ROC-AUC."""
    print("Starting evaluation...")
    
    errors = {}
    all_y_true = []
    all_y_scores = []
    
    for ticker in valid_tickers:
        if ticker not in factor_sensitivities.index:
            continue
            
        try:
            # A. Prepare Test Data
            Y_test = test_returns[ticker].shift(-1)
            X_test = build_feature_set(ticker, test_returns.index, test_factors_dict, test_realized_vol)
            
            data_test = pd.concat([Y_test, X_test], axis=1).dropna()
            
            if len(data_test) < 10:
                continue
            
            Y_test_clean = data_test.iloc[:, 0]
            X_test_clean = data_test.iloc[:, 1:]
            
            # B. Predict
            betas = factor_sensitivities.loc[ticker].values.astype(float)
            predictions = np.dot(X_test_clean, betas)
            
            # C. Calc MSE
            mse = np.mean((Y_test_clean - predictions) ** 2)
            errors[ticker] = mse
            
            # D. Collect Data for ROC-AUC
            # Binary Class: 1 if Return > 0 (Up), 0 if Return <= 0 (Down)
            # Score: The raw predicted return (higher prediction = higher confidence in 'Up')
            binary_labels = (Y_test_clean > 0).astype(int)
            
            all_y_true.extend(binary_labels)
            all_y_scores.extend(predictions)
            
        except Exception as e:
            pass

    # Global ROC-AUC Calculation
    # We calculate one score across ALL stock predictions to see global predictive power
    if all_y_true:
        roc_score = roc_auc_score(all_y_true, all_y_scores)
        print(f"\nGlobal ROC-AUC Score: {roc_score:.4f}")
        
        # Interpretation
        if roc_score > 0.5:
            print("Result: Model has predictive skill (better than random).")
        else:
            print("Result: Model is performing worse than random guessing.")
    else:
        print("No valid predictions to calculate ROC-AUC.")
        roc_score = 0.5

    return pd.Series(errors), roc_score

# --- Main Execution ---

def main():
    # 1. Load
    factors_dict, prices = load_all_data()
    
    # 2. Derive & Split
    returns, realized_vol = calculate_derived_metrics(prices)
    
    # Dictionary comp to split all factor DFs efficiently
    train_factors = {k: split_train_test(v)[0] for k, v in factors_dict.items()}
    test_factors  = {k: split_train_test(v)[1] for k, v in factors_dict.items()}
    
    train_returns, test_returns = split_train_test(returns)
    train_vol, test_vol = split_train_test(realized_vol)
    
    # 3. Intersect Tickers
    valid_tickers = get_valid_tickers(factors_dict, prices)
    print(f"Analyzing {len(valid_tickers)} tickers.")
    
    # 4. Train
    betas = train_model(valid_tickers, train_returns, train_factors, train_vol)
    betas.to_csv("Factor_Betas_Refactored.csv")
    print(betas.head())
    
    # 5. Evaluate
    mse_results, auc_score = evaluate_model(valid_tickers, test_returns, test_factors, test_vol, betas)
    mse_results.to_csv("Test_MSEs_Refactored.csv")
    
    # 6. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mse_results.index, mse_results.values)
    plt.xticks(rotation=90)
    plt.ylabel("Mean Squared Error")
    plt.title(f"Test MSE per Stock (Global ROC-AUC: {auc_score:.3f})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    