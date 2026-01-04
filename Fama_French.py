from sklearn.linear_model import LinearRegression
import pandas_datareader as pdr
import pandas as pd

# Download Fama-French factors
ff_factors = pdr.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench')[0]

sectors = pd.read_excel('SPX_sectors_data.xlsx',sheet_name='Sectors',header=0,index_col=0)


# For each sector
for sector in sectors:
    y = sector_returns[sector] - risk_free_rate
    X = ff_factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Betas represent factor exposures
    factor_exposures[sector] = model.coef_
