import numpy as np 
import pandas as pd 
import yfinance as yf 
from datetime import datetime, timedelta

stocks = [ 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'AMD', 'INTC']

end_date = datetime.now()

start_date = end_date - timedelta(days=730)

data = yf.download(stocks, start= start_date, end = end_date, progress= False)

all_data = []

for stock in stocks:
    try:
        close = data['Close'][stock].values
        volume= data['Volume'][stock].values
        dates = data.index

        valid = (np.isnan(close)==False) & (np.isnan(volume)==False) & (volume>0)
        if valid.sum() < 100:
            continue
        close = close[valid]
        volume = volume[valid]
        dates = dates[valid]
        close_today = close[:-1]
        close_tomorrow = close[1:]
        target= (close_today<close_tomorrow).astype(int)
        dates = dates[:-1]
        volume = volume[:-1]

        df = pd.DataFrame({'Date' : dates , 'Stock' : stock , 'Close' : close_today , 'Volume' : volume , 'Target' : target})
        all_data.append(df)
    except:
        continue

df_combined= pd.concat(all_data, ignore_index=True)
df_combined.to_csv("data_raw.csv", index=False)

