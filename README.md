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

print(new_cases_vector_actual_data)
# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data to include only the relevant dates
data_filtered = data[(data['Date'] >= '2021-06-01') & (data['Date'] <= '2021-08-14')]

# Retain the Date column separately
dates = data_filtered['Date']

# Normalize the time series data excluding the Date column
scaler = MinMaxScaler()
time_series_data = data_filtered.drop(columns=['Date'])
normalized_data = pd.DataFrame(scaler.fit_transform(time_series_data), columns=time_series_data.columns)

# Add the Date column back to the normalized data
normalized_data['Date'] = dates.values

# Split the data into training and testing sets
train_set = normalized_data[(normalized_data['Date'] >= '2021-06-01') & (normalized_data['Date'] <= '2021-07-17')]
test_set = normalized_data[(normalized_data['Date'] >= '2021-07-18') & (normalized_data['Date'] <= '2021-08-14')]

# Extract the 'New COVID-19 Cases' column and convert it to a time series
covid_cases_ts = data_filtered.set_index('Date')['New COVID-19 Cases']

# Define the training period explicitly using the subset function
train_data = covid_cases_ts[:'2021-07-17']

# Fit ARIMA(9,2,2) model
arima_model = ARIMA(train_data, order=(9, 2, 2)).fit()

# Print the model summary
print(arima_model.summary())

# Forecast on the test set
test_data = covid_cases_ts['2021-07-18':'2021-08-14']
forecast = arima_model.forecast(steps=len(test_data))

# Calculate performance metrics
mse = mean_squared_error(test_data, forecast)
mae = mean_absolute_error(test_data, forecast)
mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
rmse = np.sqrt(mse)
rrmse = rmse / (np.max(test_data) - np.min(test_data))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Relative Root Mean Square Error (RRMSE): {rrmse}")


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
start_date = '2021-09-12'
end_date = '2021-09-18'

rolling_forecast = []
history = list(train_data)

for date in pd.date_range(start=start_date, end=end_date):

        model = ARIMA(history, order = (9, 2, 2), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        output = model_fit.forecast()
        rolling_forecast.append(output[0])
        history.append(output[0])

rolling_forecast_arima_with_no_exo = pd.Series(rolling_forecast, index=pd.date_range(start=start_date, end=end_date))


print (rolling_forecast_arima_with_no_exo)


##### LSTM MODEL
pip install tensorflow
pip install --upgrade tensorflow keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
data = pd.read_excel("C:\\Users\\A B Siddik\\Desktop\\ARIMA\\data.xlsx")

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data to include only the relevant dates
data_filtered = data[(data['Date'] >= '2021-06-01') & (data['Date'] <= '2021-08-14')]

# Extract the 'New COVID-19 Cases' column and normalize the data
covid_cases = data_filtered['New COVID-19 Cases'].values
scaler = MinMaxScaler()
covid_cases_scaled = scaler.fit_transform(covid_cases.reshape(-1, 1))

# Define the training period explicitly
train_end_date = '2021-07-17'

test_start_date = '2021-07-18'
test_end_date = '2021-08-14'

train_data = data_filtered[data_filtered['Date'] <= train_end_date]['New COVID-19 Cases'].values
test_data = data_filtered[(data_filtered['Date'] >= test_start_date) & (data_filtered['Date'] <= test_end_date)]['New COVID-19 Cases'].values

# Normalize train and test data
train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

print(f"train_data length: {len(train_data_scaled)}, test_data length: {len(test_data_scaled)}")

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

# Define sequence length

seq_length = 10

# Create sequences for training and testing
x_train, y_train = create_sequences(train_data_scaled, seq_length)
x_test, y_test = create_sequences(test_data_scaled, seq_length)

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=200, input_shape=(seq_length, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=472, batch_size=22, verbose=1)  # Reduced epochs for quicker debugging

# Forecasting

y_pred = model.predict(x_test)
print(f"y_pred shape: {y_pred.shape}")

# Rescale the predicted and true values back to the original scale
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = scaler.inverse_transform(y_pred)

# Calculate performance metrics
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
rmse = np.sqrt(mse)
rrmse = rmse / (np.max(y_test_rescaled) - np.min(y_test_rescaled))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Relative Root Mean Square Error (RRMSE): {rrmse}")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example: Generate synthetic training data
n_samples = 100
train_data = np.random.rand(n_samples, 1)  # Example 1D training data

# Step 1: Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)

# Define sequence length
seq_length = 5  # Replace with the actual sequence length used in your model

# Prepare sequences for training
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_data_scaled, seq_length)

# Step 2: Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(seq_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=8, verbose=1)

# Step 3: Perform rolling forecasting
forecast_start_date = '2021-09-12'
forecast_end_date = '2021-09-18'
forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date)

# Initialize history with the last `seq_length` sequences from training data
history = list(train_data_scaled[-seq_length:])
history = [seq[0] for seq in history]  # Ensure it's a list of scalar values
rolling_forecast = []

for date in forecast_dates:
    input_seq = np.array(history[-seq_length:]).reshape((1, seq_length, 1))
    forecast = model.predict(input_seq)  # Ensure the `model` is defined and trained
    rolling_forecast.append(forecast[0, 0])
    history.append(forecast[0, 0])

# Rescale the forecasted values back to the original scale
rolling_forecast_rescaled = scaler.inverse_transform(np.array(rolling_forecast).reshape(-1, 1))

# Create a DataFrame for plotting
rolling_forecast_lstm_no_exo_df = pd.DataFrame(data=rolling_forecast_rescaled, index=forecast_dates, columns=['Forecast'])
rolling_forecast_lstm_no_exo_df['Forecast'] = np.abs(rolling_forecast_lstm_no_exo_df['Forecast'])

# Display the DataFrame
print(rolling_forecast_lstm_no_exo_df)

#### ARIMA+LSTM HYBRID MODEL

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_excel("C:\\Users\\A B Siddik\\Desktop\\ARIMA\\data.xlsx")

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data to include only the relevant dates
data_filtered = data[(data['Date'] >= '2021-06-01') & (data['Date'] <= '2021-08-14')]

# Extract the 'New COVID-19 Cases' column
covid_cases = data_filtered['New COVID-19 Cases'].values

# Train ARIMA model
arima_model = ARIMA(covid_cases, order=(9, 2, 2))
arima_fit = arima_model.fit()

# Get residuals from ARIMA model
residuals = arima_fit.resid

# Normalize the residuals
scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1))

# Define sequence length
seq_length = 10

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

# Create sequences for training
x_train, y_train = create_sequences(residuals_scaled, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=200, input_shape=(seq_length, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam())

# Train the model
history = model.fit(x_train, y_train, epochs=472, batch_size=22, verbose=1)

# Print the model summary
print(model.summary())


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
data = pd.read_excel("C:\\Users\\A B Siddik\\Desktop\\ARIMA\\data.xlsx")

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data to include only the relevant dates
data_filtered = data[(data['Date'] >= '2021-06-01') & (data['Date'] <= '2021-08-14')]

# Extract the 'New COVID-19 Cases' column
covid_cases = data_filtered['New COVID-19 Cases'].values

# Split the data into training and test sets
train_end_date = '2021-07-17'
test_start_date = '2021-07-18'

train_data = data_filtered[data_filtered['Date'] <= train_end_date]['New COVID-19 Cases'].values
test_data = data_filtered[data_filtered['Date'] >= test_start_date]['New COVID-19 Cases'].values

# Train ARIMA model
arima_model = ARIMA(train_data, order=(9, 2, 2))
arima_fit = arima_model.fit()

# Get residuals from ARIMA model
train_residuals = arima_fit.resid

# Normalize the residuals
scaler = MinMaxScaler()
train_residuals_scaled = scaler.fit_transform(train_residuals.reshape(-1, 1))

# Define sequence length
seq_length = 10

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

# Create sequences for training
x_train, y_train = create_sequences(train_residuals_scaled, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=200, input_shape=(seq_length, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam())

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=22, verbose=1)

# Predict the test data with ARIMA
arima_forecast = arima_fit.forecast(steps=len(test_data))
arima_residuals = test_data - arima_forecast

# Normalize the ARIMA residuals
arima_residuals_scaled = scaler.transform(arima_residuals.reshape(-1, 1))

# Create sequences for LSTM predictions
x_test, y_test = create_sequences(arima_residuals_scaled, seq_length)

# Predict the residuals with LSTM
lstm_forecast_scaled = model.predict(x_test)

# Rescale the LSTM predictions
lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled)

# Combine ARIMA forecast and LSTM residual forecast
final_forecast = arima_forecast[seq_length:] + lstm_forecast.flatten()

# Calculate performance metrics
mse = mean_squared_error(test_data[seq_length:], final_forecast)
mae = mean_absolute_error(test_data[seq_length:], final_forecast)
mape = np.mean(np.abs((test_data[seq_length:] - final_forecast) / test_data[seq_length:])) * 100
rmse = np.sqrt(mse)
rrmse = rmse / (np.max(test_data[seq_length:]) - np.min(test_data[seq_length:]))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Relative Root Mean Square Error (RRMSE): {rrmse}")


import pandas as pd
import numpy as np  # Ensure numpy is also imported if it's being used

# Perform rolling forecasting between September 12, 2021, and September 18, 2021
forecast_start_date = '2021-09-12'
forecast_end_date = '2021-09-18'
forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date)

# Initialize history with the last `seq_length` sequences from training data
history_cases = list(train_data[-seq_length:])
history_residuals = list(train_residuals_scaled[-seq_length:])
history_residuals = [seq[0] for seq in history_residuals]  # Ensure it's a list of scalar values

rolling_forecast = []

for date in forecast_dates:
    # Use the ARIMA model to forecast the next value
    arima_forecast_next = arima_fit.forecast(steps=1)[0]
    history_cases.append(arima_forecast_next)
    history_cases = history_cases[-seq_length:]  # Keep the length of history consistent

    # Calculate the residuals
    arima_residual_next = arima_forecast_next - (history_cases[-2] if len(history_cases) > 1 else arima_forecast_next)
    history_residuals.append(scaler.transform([[arima_residual_next]])[0][0])
    history_residuals = history_residuals[-seq_length:]  # Keep the length of history consistent

    # Predict the residuals with LSTM
    input_seq = np.array(history_residuals).reshape((1, seq_length, 1))
    lstm_forecast_next_scaled = model.predict(input_seq)
    lstm_forecast_next = scaler.inverse_transform(lstm_forecast_next_scaled)[0, 0]

    # Combine ARIMA forecast and LSTM residual forecast
    final_forecast_next = arima_forecast_next + lstm_forecast_next
    rolling_forecast.append(final_forecast_next)

# Rescale the forecasted values back to the original scale
rolling_forecast_rescaled = scaler.inverse_transform(np.array(rolling_forecast).reshape(-1, 1))

# Create a DataFrame for plotting
rolling_forecast_combined_no_exo_df = pd.DataFrame(data=rolling_forecast_rescaled, index=forecast_dates, columns=['Forecast'])
rolling_forecast_combined_no_exo_df['Forecast'] = np.abs(rolling_forecast_combined_no_exo_df['Forecast'])

# Display the DataFrame
print(rolling_forecast_combined_no_exo_df)




