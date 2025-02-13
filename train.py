import pandas as pd
import numpy as np
from ib_insync import IB, Stock
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pickle
import datetime
import json

# Connect to Interactive Brokers
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
symbols = ["SPY", "AAPL", "NVDA", "MSFT", "AMZN", "META", "TSLA", "GOOGL", "AVGO", "BRK.B"]

def fetch_data_in_chunks(symbol, total_years=8, chunk_years=1, bar_size='1 day', useRTH = True):
    """
    Fetch historical stock data in chunks.

    Args:
        symbol (str): The stock symbol to fetch data for.
        total_years (int): Total years of data to fetch.
        chunk_years (int): The number of years in each chunk.
        bar_size (str): The size of the bars for historical data.

    Returns:
        list: A list of DataFrames containing the historical data.
    """
    all_data = []
    current_date = datetime.datetime.today()
    start_date = current_date - datetime.timedelta(days=total_years * 365)

    for start_year in range(0, total_years, chunk_years):
        end_date = start_date + datetime.timedelta(days=(start_year + chunk_years) * 365)
        query_time = end_date.strftime("%Y%m%d %H:%M:%S")
        duration = f"{chunk_years} Y"
        
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)

        bars = ib.reqHistoricalData(
            contract,
            endDateTime=query_time,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=useRTH
        )

        df = pd.DataFrame(bars)
        df['symbol'] = symbol
        all_data.append(df)

    return all_data

def fetch_data():
    """
    Fetch daily and intraday stock data for a predefined list of symbols.

    Returns:
        tuple: A tuple containing two lists:
            - dataframes_daily: List of DataFrames with daily stock data.
            - dataframes_intraday: List of DataFrames with intraday stock data.
    """

    dataframes_daily = []
    dataframes_intraday = []

    for symbol in symbols:
        try:
            # Fetch daily data
            daily_df = fetch_data_in_chunks(symbol)
            dataframes_daily.extend(daily_df)
            print(f"----------------------------------Fetched daily data for {symbol}----------------------------------------------")

            # Fetch intraday data
            intraday_df = fetch_data_in_chunks(symbol, bar_size='30 mins', useRTH = False)
            dataframes_intraday.extend(intraday_df)
            print(f"----------------------------------Fetched intraday data for {symbol}-------------------------------------------")

        except Exception as e:
            print(f"---------------------------------Error fetching data for {symbol}: {e}------------------------------------------")

    print("---------------------------------------Successfully finished fetching data.-------------------------------------------")
    ib.disconnect()
    return dataframes_daily, dataframes_intraday

# Fetch data
dataframes_daily, dataframes_intraday = fetch_data()


daily_json = [df.to_json(orient='records', date_format='iso') for df in dataframes_daily]
intraday_json = [df.to_json(orient='records', date_format='iso') for df in dataframes_intraday]

# Write JSON data to files
with open('daily_data.json', 'w') as f:
    json.dump(daily_json, f)
    print("----------------------------------------Successfully saved fetched daily data as Json.-----------------------------------------")

with open('intraday_data.json', 'w') as f:
    json.dump(intraday_json, f)
    print("---------------------------------------Successfully saved fetched intraday data as Json.--------------------------------------")


# Combine and preprocess daily data
daily_data = pd.concat(dataframes_daily)
daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data['date'] = daily_data['date'] + pd.Timedelta(hours=9)
daily_data['date'] = daily_data['date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
daily_data.set_index('date', inplace=True)

# Combine and preprocess intraday data
intraday_data = pd.concat(dataframes_intraday)
intraday_data['date'] = pd.to_datetime(intraday_data['date']).dt.tz_convert('America/New_York')
intraday_data.set_index('date', inplace=True)

# Pre-Market and Gap Calculations
pre_market_start = pd.Timestamp('04:00:00').time()
pre_market_end = pd.Timestamp('09:00:00').time()
intraday_data['is_pre_market'] = intraday_data.index.map(
    lambda x: pre_market_start <= x.time() <= pre_market_end
)

daily_data['next_open'] = daily_data.groupby('symbol')['open'].shift(-1)
daily_data['gap_pct'] = ((daily_data['next_open'] - daily_data['close']) / daily_data['close']) * 100

# Pre-Market Volume and Dollar Volume Calculations
intraday_data.reset_index(inplace=True)
intraday_data['date'] = intraday_data['date'].dt.date

pre_market_stats = intraday_data[intraday_data['is_pre_market']].groupby(['date', 'symbol']).agg(
    pre_market_volume=('volume', 'sum'),
    pre_market_dollar_vol=('volume', lambda x: (x * intraday_data.loc[x.index, 'close']).sum())
).reset_index()

daily_data.reset_index(inplace=True)
daily_data['date'] = daily_data['date'].dt.date
combined_data = daily_data.merge(pre_market_stats, on=['date', 'symbol'], how='left')

# Relative Volume Calculation
combined_data['avg_pre_market_vol_30d'] = combined_data.groupby('symbol')['pre_market_volume'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)
combined_data['avg_pre_market_dollar_vol_30d'] = combined_data.groupby('symbol')['pre_market_dollar_vol'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)

combined_data['rel_vol'] = combined_data['pre_market_volume'] / combined_data['avg_pre_market_vol_30d']
combined_data['rel_dollar_vol'] = combined_data['pre_market_dollar_vol'] / combined_data['avg_pre_market_dollar_vol_30d']

combined_data['gap_fill'] = (combined_data['low'] <= combined_data['close'].shift(1)).astype(int)
# Prepare data for modeling
feature_cols = ['rel_vol', 'rel_dollar_vol', 'gap_pct']
X = combined_data[feature_cols].fillna(0)
y = combined_data['gap_fill'].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance visualization
coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_[0]})
coef_df.sort_values(by='Coefficient', ascending=False, inplace=True)
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')
plt.show()

# Save the model and processed data
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training completed and saved.")
print("------------------------finish-------------------------------")