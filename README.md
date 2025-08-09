# Forecasting-project
### ARIMA MODEL

pip install statsmodels
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install openpyxl

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_excel("C:\\Users\\A B Siddik\\Desktop\\ARIMA\\data.xlsx")
data.head()
# Filter the data to include only the relevant dates
filtered_data = data[(data['Date'] >= '2021-09-12') & (data['Date'] <= '2021-09-18')]

# Extract the 'New COVID-19 Cases' column
new_cases_vector_actual_data = filtered_data['New COVID-19 Cases'].values

print(new_cases_vector_actual_dat

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# === Step 1: Load and preprocess data ===
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Filter dataset between Jan 4, 2021, and Sep 18, 2021
data_filtered = data[(data['Date'] >= '2021-01-04') & (data['Date'] <= '2021-09-18')].reset_index(drop=True)

# Keep original Date column separately
dates = data_filtered['Date']

# Normalize all columns except Date
scaler = MinMaxScaler()
features = data_filtered.drop(columns=['Date'])
normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
normalized_features['Date'] = dates

# === Step 2: Split into train/val/test ===
train_set = normalized_features[(normalized_features['Date'] >= '2021-01-04') & (normalized_features['Date'] <= '2021-07-02')]
val_set = normalized_features[(normalized_features['Date'] >= '2021-07-03') & (normalized_features['Date'] <= '2021-08-10')]
test_set = normalized_features[(normalized_features['Date'] >= '2021-08-11') & (normalized_features['Date'] <= '2021-09-18')]

# Reset index for each
train_set = train_set.reset_index(drop=True)
val_set = val_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)

# === Step 3: Time series model on actual cases ===
# We'll use unnormalized actual cases for ARIMA forecasting
covid_cases_ts = data_filtered.set_index('Date')['New COVID-19 Cases']  # Replace 'Cases' with your actual column name

# Train ARIMA model on training + validation set
arima_train_series = covid_cases_ts['2021-01-04':'2021-08-10']
arima_model = ARIMA(arima_train_series, order=(6,1,6))  # You can optimize (p,d,q) or use auto_arima
arima_model = arima_model.fit()

# Print the model summary
print(arima_model.summary())
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

# Step 4: Forecast and evaluate
test_data = covid_cases_ts['2021-08-11':'2021-09-18']
forecast = arima_model.forecast(steps=len(test_data))
forecast = pd.Series(forecast, index=test_data.index)

# Metrics
mse = mean_squared_error(test_data, forecast)
mae = mean_absolute_error(test_data, forecast)
mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
rmse = np.sqrt(mse)
rrmse = rmse / (np.max(test_data) - np.min(test_data))
nrmse = rmse / np.mean(test_data)  # Normalized RMSE by mean
r2 = r2_score(test_data, forecast)

# Print results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Relative RMSE (RRMSE): {rrmse:.4f}")
print(f"Normalized RMSE (NRMSE): {nrmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Initialize history with actual training values (no datetime index)
history = list(arima_train_series.values)

start_date = '2021-09-12'
end_date = '2021-09-18'

rolling_forecast = []

for date in pd.date_range(start=start_date, end=end_date):
    model = ARIMA(history, order=(6, 1, 6), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    output = model_fit.forecast(steps=1)  # output is np.ndarray
    next_value = output[0]                # fixed line
    rolling_forecast.append(next_value)
    
    history.append(next_value)            # add predicted value to history

# Convert to Series with date index
forecast_index = pd.date_range(start=start_date, end=end_date)
rolling_forecast_arima_with_no_exo = pd.Series(rolling_forecast, index=forecast_index)

# Optional: print or plot
print(rolling_forecast_arima_with_no_exo)
import matplotlib.pyplot as plt
import pandas as pd

# Define the dates
dates = pd.date_range(start='2021-09-12', end='2021-09-18')

# Define the vectors
new_cases_vector_actual_data = [19550, 19198, 16073, 15669, 19495, 18815,17577]
forecast_vector_arima_no_exo = [17695.577209, 18454.495591, 19303.898229, 19389.129789, 18751.381991, 17731.829261, 17116.208056]
# Plot the data
plt.figure(figsize=(14, 8))
plt.plot(dates, new_cases_vector_actual_data, label='Actual active cases', marker='o')
plt.plot(dates, forecast_vector_arima_no_exo, label='ARIMA prediction ', marker='o')


# Add title and labels
plt.title('New active cases: Actual vs. Forecasted')
plt.xlabel('Date')
plt.ylabel('Number of new cases')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()



##### LSTM MODEL


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# === Load and preprocess data ===
data = pd.read_excel("C:\\Users\\A B Siddik\\Desktop\\ARIMA\\data.xlsx")
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data = data[(data['Date'] >= '2021-01-04') & (data['Date'] <= '2021-09-18')].reset_index(drop=True)

target_col = 'New COVID-19 Cases'
target_series = data[[target_col, 'Date']].copy()

# === Define date-based splits (same as ARIMA) ===
train_data = target_series[(target_series['Date'] <= '2021-08-10')]
test_data = target_series[(target_series['Date'] >= '2021-08-11') & (target_series['Date'] <= '2021-09-18')]

# === Normalize based on training only ===
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[[target_col]])
test_scaled = scaler.transform(test_data[[target_col]])

# === Create sequences ===
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_length = 10
x_train, y_train = create_sequences(train_scaled, seq_length)
x_test, y_test = create_sequences(test_scaled, seq_length)

# === Build and train LSTM model ===
model = Sequential()
model.add(LSTM(64, input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=200, batch_size=16, verbose=1)

# === Predict on test set ===
y_pred = model.predict(x_test)

# === Inverse scale predictions and true values ===
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# === Evaluate LSTM model ===
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
rmse = np.sqrt(mse)
rrmse = rmse / (np.max(y_test_rescaled) - np.min(y_test_rescaled))
nrmse = rmse / np.mean(y_test_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print("\n📊 Final LSTM Forecast Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Relative RMSE (RRMSE): {rrmse:.4f}")
print(f"Normalized RMSE (NRMSE): {nrmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
# === Define forecast horizon (same as your ARIMA example) ===
forecast_horizon = pd.date_range(start='2021-09-12', end='2021-09-18')

# === Start with the last known sequence from test data ===
last_sequence = test_scaled[-seq_length:]  # shape (10, 1)

forecast_scaled = []

for _ in range(len(forecast_horizon)):
    # Reshape to (1, seq_len, 1) for LSTM input
    input_seq = last_sequence.reshape((1, seq_length, 1))

    # Predict next value
    next_scaled = model.predict(input_seq, verbose=0)[0][0]

    # Append prediction
    forecast_scaled.append([next_scaled])

    # Update sequence: drop first, append prediction
    last_sequence = np.vstack((last_sequence[1:], [[next_scaled]]))

# Inverse transform forecasted values
forecast_scaled = np.array(forecast_scaled)
forecast_rescaled = scaler.inverse_transform(forecast_scaled)

# Convert to Pandas Series with date index
lstm_forecast_series = pd.Series(forecast_rescaled.flatten(), index=forecast_horizon)

print("\n📈 LSTM Rolling Forecast:")
print(lstm_forecast_series)
import matplotlib.pyplot as plt
import pandas as pd

# Define the dates
dates = pd.date_range(start='2021-09-12', end='2021-09-18')

# Define the vectors
new_cases_vector_actual_data = [19550, 19198, 16073, 15669, 19495, 18815, 17577]
forecast_vector_lstm_no_exo = [18444.546875, 19154.939453, 19460.316406, 19636.972656, 19800.009766, 19999.65625,  20213.628906]

# Plot the data
plt.figure(figsize=(14, 8))
plt.plot(dates, new_cases_vector_actual_data, label='Actual New Cases', marker='o')
plt.plot(dates, forecast_vector_lstm_no_exo, label='LSTM prediction', marker='o')


# Add title and labels
plt.title('Active Cases: Actual vs. Forecasted')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()



#### ARIMA+LSTM HYBRID MODEL

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt

# === Load data ===
data = pd.read_excel("C:\\Users\\A B Siddik\\Desktop\\ARIMA\\data.xlsx")
data['Date'] = pd.to_datetime(data['Date'])
data = data[(data['Date'] >= '2021-01-04') & (data['Date'] <= '2021-09-18')].reset_index(drop=True)

# === Extract series ===
target_col = 'New COVID-19 Cases'
target_series = data[target_col].values
dates = data['Date']

# === Split ARIMA train/test ===
train_end = '2021-08-10'
test_start = '2021-08-11'
test_end = '2021-09-18'
train_arima = data[(data['Date'] <= train_end)][target_col]
test_arima = data[(data['Date'] >= test_start) & (data['Date'] <= test_end)][target_col]

# === Fit ARIMA with fixed order (6, 1, 6) ===
order = (6, 1, 6)
arima_model = ARIMA(train_arima, order=order).fit()
arima_forecast = arima_model.forecast(steps=len(test_arima))
arima_forecast.index = test_arima.index

# === Get residuals ===
arima_pred_train = arima_model.predict(start=1, end=len(train_arima)-1)
actual_train = train_arima[1:]
residuals = actual_train.values - arima_pred_train.values

# === Normalize residuals ===
scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1))

# === Create LSTM sequences ===
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_length = 15
x_lstm, y_lstm = create_sequences(residuals_scaled, seq_length)

# === Build LSTM model ===
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_lstm, y_lstm, epochs=200, batch_size=16, verbose=1)

# === Forecast LSTM residuals ===
resid_history = residuals_scaled[-seq_length:].flatten().tolist()
lstm_forecast_scaled = []

for _ in range(len(test_arima)):
    input_seq = np.array(resid_history[-seq_length:]).reshape(1, seq_length, 1)
    pred = model.predict(input_seq, verbose=0)[0][0]
    lstm_forecast_scaled.append(pred)
    resid_history.append(pred)

lstm_forecast_resid = scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1, 1)).flatten()

# === Combine ARIMA and LSTM ===
hybrid_forecast = arima_forecast.values + lstm_forecast_resid
actual = test_arima.values

# === Evaluate ===
mse = mean_squared_error(actual, hybrid_forecast)
mae = mean_absolute_error(actual, hybrid_forecast)
mape = np.mean(np.abs((actual - hybrid_forecast) / actual)) * 100
rmse = np.sqrt(mse)
rrmse = rmse / (np.max(actual) - np.min(actual))
nrmse = rmse / np.mean(actual)
r2 = r2_score(actual, hybrid_forecast)

print("\n\U0001F4CA Final Hybrid ARIMA + LSTM Forecast Evaluation:")
print(f"ARIMA Order Used: {order}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Relative RMSE (RRMSE): {rrmse:.4f}")
print(f"Normalized RMSE (NRMSE): {nrmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
# === Step 7: Rolling Forecast with Hybrid Model (Sept 12–18) ===
forecast_start_date = '2021-09-12'
forecast_end_date = '2021-09-18'
forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date)

# Initialize history (raw cases)
history_cases = train_arima.tolist()  # from start to 2021-08-10
resid_history = residuals_scaled[-seq_length:].flatten().tolist()

rolling_forecast = []

for _ in forecast_dates:
    # --- ARIMA Forecast ---
    temp_arima = ARIMA(history_cases, order=(6, 1, 6))
    temp_fit = temp_arima.fit()
    arima_next = temp_fit.forecast(steps=1)[0]
    history_cases.append(arima_next)

    # --- Residual Prediction with LSTM ---
    input_seq = np.array(resid_history[-seq_length:]).reshape(1, seq_length, 1)
    lstm_scaled_next = model.predict(input_seq, verbose=0)
    lstm_resid = scaler.inverse_transform(lstm_scaled_next)[0][0]
    resid_history.append(lstm_scaled_next[0][0])

    # --- Combine ARIMA + LSTM ---
    final_forecast = arima_next + lstm_resid
    rolling_forecast.append(final_forecast)

# === Step 8: Output Forecast DataFrame ===
hybrid_forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecast': np.clip(np.round(rolling_forecast), a_min=0, a_max=None).astype(int)
})
hybrid_forecast_df.set_index('Date', inplace=True)

# === Display Forecast ===
print("\n📅 Hybrid Forecast (2021-09-12 to 2021-09-18):")
print(hybrid_forecast_df)

import matplotlib.pyplot as plt
import pandas as pd

# Define the dates
dates = pd.date_range(start='2021-09-12', end='2021-09-18')

# Define the vectors
new_cases_vector_actual_data = [19550, 19198, 16073, 15669, 19495, 18815, 17577]
forecast_vector_combined_no_exo =[17806, 18832, 19763, 20292, 20106, 19318, 18672]
# Plot the data
plt.figure(figsize=(14, 8))
plt.plot(dates, new_cases_vector_actual_data, label='Actual New Cases', marker='o')
plt.plot(dates, forecast_vector_combined_no_exo, label='Hybrid ARIMA-LSTM prediction', marker='o')


# Add title and labels
plt.title('Active Cases: Actual vs. Forecasted')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()








