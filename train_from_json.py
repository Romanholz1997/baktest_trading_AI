import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pickle
import json
# Load JSON files
with open('daily_data.json', 'r') as f:
    daily_json = json.load(f)

with open('intraday_data.json', 'r') as f:
    intraday_json = json.load(f)

# Convert JSON back into DataFrame
dataframes_daily = [pd.DataFrame(json.loads(json_str)) for json_str in daily_json]
dataframes_intraday = [pd.DataFrame(json.loads(json_str)) for json_str in intraday_json]


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
combined_data['gap_fill'] = ((combined_data['low'] <= combined_data['close'].shift(1))).astype(int)
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