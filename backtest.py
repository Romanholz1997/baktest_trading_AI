import pandas as pd
from ib_insync import IB, Stock
import matplotlib.pyplot as plt
import pickle
import json
import numpy as np
import quantstats as qs
import datetime

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
symbols = ["SPY", "AAPL", "NVDA", "MSFT", "AMZN", "META", "TSLA", "GOOGL", "AVGO", "BRK.B"]

# Load trained logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

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

def prepare_backtest(daily_json, intraday_json):

    # Combine and preprocess daily data
    daily_data = pd.concat(daily_json)
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data['date'] = daily_data['date'] + pd.Timedelta(hours=9)
    daily_data['date'] = daily_data['date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    daily_data.set_index('date', inplace=True)

    # Combine and preprocess intraday data
    intraday_data = pd.concat(intraday_json)
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
    intraday_data['time'] = intraday_data['date']
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

    combined_data['date'] = pd.to_datetime(combined_data['date']).dt.date
    combined_data.set_index('date', inplace=True)

    intraday_data['time'] = pd.to_datetime(intraday_data['time'])
    print("Backtest preparing completed.")
    return combined_data, intraday_data

def backtest_strategy(data, analysis_data, model):
    """
    Backtesting a simple long-short strategy based on gap-fill predictions.

    - Go long if gap fill is predicted (1)
    - Go short if gap fill is not predicted (0)
    - Exit at the next day's close

    Args:
        data (DataFrame): Processed stock data
        model: Trained machine learning model
    
    Returns:
        DataFrame: Strategy performance with returns
    """

    feature_cols = ['rel_vol', 'rel_dollar_vol', 'gap_pct']
    data['predicted'] = model.predict(data[feature_cols].fillna(0))

    # Define strategy returns
    data['next_close'] = data.groupby('symbol')['close'].shift(-1)
    data['previous_close'] = data.groupby('symbol')['close'].shift(1)  # Create a new column for previous close
    data['model_probability'] = model.predict_proba(data[feature_cols].fillna(0))[:, 1]  # Get 

    filtered_data = data[(data['model_probability'] >= 0.7)].copy()
    filtered_data = filtered_data[filtered_data['symbol'] == 'SPY']
    trades = []
    last_trade_date = None  # Variable to track the last trade date
    analysis_data = analysis_data[analysis_data['symbol'] == 'SPY'].copy()
    # Now you can safely modify it without warnings
    analysis_data.loc[:, 'current_price'] = analysis_data['close']
    # Create a merged DataFrame with relevant analysis data
    merged_data = filtered_data.merge(analysis_data, left_index=True, right_on='date', suffixes=('', '_analysis'))
    # Iterate through the merged DataFrame
    for index, row in merged_data.iterrows():
        index = pd.to_datetime(row['time'])
        if last_trade_date is not None and last_trade_date.date() == index.date():
            continue  # Skip to the next iteration if we already traded today

        entry_price = row['open']
        target_price = row['previous_close']
        stop_loss = entry_price - (target_price - entry_price)

        # Check for target hit, stop-loss hit, or market close condition
        current_price = row['current_price']
        if entry_price > target_price: # sell
            if current_price <= target_price:
                trades.append({'date': index, 'symbol': row['symbol'], 'entry': entry_price, 'exit': target_price, 'profit': entry_price - target_price })
                last_trade_date = index  # Update the last trade date
            elif current_price >= stop_loss:
                trades.append({'date': index, 'symbol': row['symbol'], 'entry': entry_price, 'exit': stop_loss, 'profit': entry_price - stop_loss})
                last_trade_date = index  # Update the last trade date
            elif row['time'].hour == 16 and row['time'].minute == 0:  # Market closes at 4:00 PM
                trades.append({'date': index, 'symbol': row['symbol'], 'entry': entry_price, 'exit': current_price, 'profit': entry_price - current_price })
                last_trade_date = index  # Update the last trade date
        else :  # buy
            if current_price >= target_price:
                trades.append({'date': index, 'symbol': row['symbol'], 'entry': entry_price, 'exit': target_price, 'profit': target_price - entry_price})
                last_trade_date = index  # Update the last trade date
            elif current_price <= stop_loss:
                trades.append({'date': index, 'symbol': row['symbol'], 'entry': entry_price, 'exit': stop_loss, 'profit': stop_loss - entry_price})
                last_trade_date = index  # Update the last trade date
            elif row['time'].hour == 16 and row['time'].minute == 0:  # Market closes at 4:00 PM
                trades.append({'date': index, 'symbol': row['symbol'], 'entry': entry_price, 'exit': current_price, 'profit': current_price - entry_price})
                last_trade_date = index  # Update the last trade date
    
    trades = pd.DataFrame(trades)
    trades['date'] = pd.to_datetime(trades['date']).dt.date
    trades['strategy_return'] = trades['profit'] / trades['entry']
    trades.set_index('date', inplace=True)
    data = data[data['symbol'] == 'SPY']
    combined = data.merge(trades, on=['date', 'symbol'], how='left')
    combined.reset_index(inplace=True)
    combined['date'] = pd.to_datetime(combined['date'])

    combined.to_csv('backtest_results.csv', index=False)
    print("Successfully saved csv file as 'backtest_results.csv'.")
    combined.dropna(subset=['strategy_return'], inplace=True)
    return combined

try:
    dataframes_daily, dataframes_intraday = fetch_data()
    combined_data, intraday_data = prepare_backtest(dataframes_daily, dataframes_intraday)
    backtest_results = backtest_strategy(combined_data, intraday_data, model)
    returns = backtest_results.groupby('date')['strategy_return'].mean()

    # Convert returns to cumulative performance
    cumulative_returns = (1 + returns).cumprod()

    # Run Quantstats Performance Analysis
    qs.extend_pandas()
    qs.reports.html(cumulative_returns, output="backtest_report.html")

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    cumulative_returns.plot(title="Strategy Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid()
    plt.show()

    print("Backtesting completed. Performance report saved as 'backtest_report.html' and 'backtest_results.csv'.")

except Exception as e:
    print(f"An error occurred: {e}")