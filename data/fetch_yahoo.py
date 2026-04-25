import yfinance as yf, pandas as pd, numpy as np, os
from datetime import datetime, timedelta

def fetch_yahoo_data(ticker='SPY', years=5,
                     output_path='data/spy_prices.csv'):
    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=False, progress=False)
                     
    if df.empty: raise ValueError('No data returned')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    # Check if 'adj close' is present to perform auto_adjust manually or skip
    if 'adj close' in df.columns:
        df['close'] = df['adj close']
        
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.index.name = 'date'
    df = df.reset_index()
    df = df.ffill().bfill()               # fill any NaNs
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Saved {len(df)} rows to {output_path}')
    return df

if __name__ == '__main__': fetch_yahoo_data()
