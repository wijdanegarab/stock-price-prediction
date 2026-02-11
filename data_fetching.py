import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta







stocks = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AVGO', 'COST', 'ASML',
    'NFLX', 'ADOBE', 'INTC', 'AMD', 'CSCO', 'PYPL', 'ADBE', 'NFLX', 'CRM', 'DDOG',
    'SHOP', 'SNOW', 'CRWD', 'ZS', 'NET', 'OKTA', 'SPLK', 'MSTR', 'ROKU', 'ABNB',
    'DASH', 'SPOT', 'UBER', 'LYFT', 'COIN', 'MARA', 'RIOT', 'MARA', 'HOOD', 'IPO',
    'SQ', 'UPST', 'SOFI', 'ACHR', 'RBLX', 'AI', 'PLTR', 'CLOV', 'WISH', 'CLNE'
]

stocks = list(set(stocks))
print(f"\nSelected {len(stocks)} stocks")
print(f"Stocks: {', '.join(stocks[:10])}... and {len(stocks)-10} more")





end_date = datetime.now()
start_date = end_date - timedelta(days=730)

print(f"\nDate range: {start_date.date()} to {end_date.date()}")
print(f"Downloading from: https://finance.yahoo.com")


data = yf.download(
    stocks,
    start=start_date,
    end=end_date,
    progress=False
)

print(f"✓ Data downloaded successfully!")
print(f"  Shape: {data.shape}")






print("PREPARING DATA")


all_data = []
successful_stocks = 0

for stock in stocks:
    try:
        if len(stocks) == 1:
            close = data['Close'].values
            volume = data['Volume'].values
            dates = data.index
        else:
            close = data['Close'][stock].values
            volume = data['Volume'][stock].values
            dates = data.index
        

        valid_mask = ~(np.isnan(close) | np.isnan(volume) | (volume == 0))
        
        if valid_mask.sum() < 100:
            continue
        
        close = close[valid_mask]
        volume = volume[valid_mask]
        dates = dates[valid_mask]
        
        

        
        close_tomorrow = close[1:]
        close_today = close[:-1]
        target = (close_tomorrow > close_today).astype(int)
        
        dates = dates[:-1]
        close_today = close_today
        volume = volume[:-1]
        
        df = pd.DataFrame({
            'Date': dates,
            'Stock': stock,
            'Close': close_today,
            'Volume': volume,
            'Target': target
        })
        
        all_data.append(df)
        successful_stocks += 1
        
        up_count = target.sum()
        down_count = len(target) - up_count
        print(f"✓ {stock:6s} - {len(df):3d} days | UP: {up_count:3d} | DOWN: {down_count:3d}")
        
    except Exception as e:
        print(f"✗ {stock:6s} - Error: {str(e)[:40]}")
        continue






print("COMBINING DATA")


df_combined = pd.concat(all_data, ignore_index=True)

print(f"Total records: {len(df_combined)}")
print(f"Unique stocks: {df_combined['Stock'].nunique()}")
print(f"Date range: {df_combined['Date'].min().date()} to {df_combined['Date'].max().date()}")


target_count = df_combined['Target'].value_counts()
print(f"\nTarget distribution:")
print(f"  DOWN (0): {target_count.get(0, 0):6d} ({target_count.get(0, 0)/len(df_combined)*100:.1f}%)")
print(f"  UP   (1): {target_count.get(1, 0):6d} ({target_count.get(1, 0)/len(df_combined)*100:.1f}%)")





print("SAVING DATA")


df_combined.to_csv("data_raw.csv", index=False)
print("✓ Raw data saved: data_raw.csv")


summary = df_combined.groupby('Stock').agg({
    'Close': ['min', 'max', 'mean'],
    'Volume': 'mean',
    'Target': ['sum', 'count']
}).round(2)

summary.to_csv("data_summary.csv")
print("✓ Summary saved: data_summary.csv")


print("done!")

