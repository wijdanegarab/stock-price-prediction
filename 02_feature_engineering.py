import numpy as np
import pandas as pd




df = pd.read_csv("data_raw.csv")
df['Date'] = pd.to_datetime(df['Date'])



features_list = []

for stock in df['Stock'].unique():
    stock_df = df[df['Stock'] == stock].sort_values('Date').reset_index(drop=True).copy()
    
    if len(stock_df) < 50:
        continue
    
   
    prices = stock_df['Close'].values
    volume = stock_df['Volume'].values
    n = len(prices)
    
   
    features = stock_df[['Date', 'Stock', 'Close', 'Volume', 'Target']].copy()
    

    features['SMA_20'] = pd.Series(prices).rolling(20).mean().values
    features['SMA_50'] = pd.Series(prices).rolling(50).mean().values
    
  
    features['EMA_12'] = pd.Series(prices).ewm(span=12).mean().values
    features['EMA_26'] = pd.Series(prices).ewm(span=26).mean().values
    
  
    delta = pd.Series(prices).diff()
    gain = delta.copy()
    gain[gain < 0] = 0
    loss = -delta.copy()
    loss[loss < 0] = 0
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    features['RSI_14'] = (100 - (100 / (1 + rs))).values
    
   
    ema12 = pd.Series(prices).ewm(span=12).mean()
    ema26 = pd.Series(prices).ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['MACD'] = macd.values
    features['MACD_Signal'] = signal.values
    features['MACD_Histogram'] = (macd - signal).values
    
 
    sma20 = pd.Series(prices).rolling(20).mean()
    std20 = pd.Series(prices).rolling(20).std()
    features['BB_Upper'] = (sma20 + 2*std20).values
    features['BB_Middle'] = sma20.values
    features['BB_Lower'] = (sma20 - 2*std20).values
    features['BB_Width'] = (4*std20).values
    features['BB_Position'] = ((pd.Series(prices) - (sma20 - 2*std20)) / (4*std20 + 1e-8)).values
    

    features['Volatility_20'] = pd.Series(prices).rolling(20).std().values
    features['Volatility_50'] = pd.Series(prices).rolling(50).std().values
    
  
    features['Momentum_10'] = (prices - np.roll(prices, 10)).astype(float)
    features['Momentum_20'] = (prices - np.roll(prices, 20)).astype(float)
    
 
    returns_1d = pd.Series(prices).pct_change()
    returns_5d = pd.Series(prices).pct_change(5)
    features['Returns_1d'] = returns_1d.values
    features['Returns_5d'] = returns_5d.values
    

    features['Volume_SMA_20'] = pd.Series(volume).rolling(20).mean().values
    features['Volume_Ratio'] = (pd.Series(volume) / (pd.Series(volume).rolling(20).mean() + 1e-8)).values
    features['Price_Range_20'] = (pd.Series(prices).rolling(20).max() - pd.Series(prices).rolling(20).min()).values
    features_list.append(features)
    
df_features = pd.concat(features_list, ignore_index=True)
df_features = df_features.fillna(method='ffill').fillna(method='bfill')
df_features = df_features.dropna()
df_features.to_csv("data_features.csv", index=False)


