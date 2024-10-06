# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # Corrected import
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Load and clean the dataset
data_cleaned = pd.read_csv(r'C:\Users\Kasutaja\Desktop\international-airline-passengers.csv', skiprows=1)

# Rename columns for simplicity
data_cleaned.columns = ['Month', 'Passengers']

# Convert the 'Month' column to a datetime format and clean any invalid data
data_cleaned['Month'] = pd.to_datetime(data_cleaned['Month'], errors='coerce')

# Drop any rows where the conversion to datetime failed
data_cleaned = data_cleaned.dropna(subset=['Month'])

# Step 2: Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(data_cleaned['Passengers'], label='Passengers')
plt.title('International Airline Passengers Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

# Step 3: Decompose the time series
decomposition = seasonal_decompose(data_cleaned['Passengers'], model='additive', period=12)

# Plot the decomposition
plt.figure(figsize=(10, 8))
decomposition.plot()
plt.show()

# Step 4: Train-test split
train_size = int(len(data_cleaned) * 0.8)
train, test = data_cleaned[:train_size], data_cleaned[train_size:]

# Step 5: ARIMA Model
model = ARIMA(train['Passengers'], order=(5, 1, 0))  # p, d, q values may be tuned
model_fit = model.fit()

# Step 6: Forecast for the test set
forecast = model_fit.forecast(steps=len(test))

# Step 7: Evaluate the model
mse = mean_squared_error(test['Passengers'], forecast)
mape = np.mean(np.abs(forecast - test['Passengers']) / np.abs(test['Passengers'])) * 100

# Step 8: Forecast Future Values
future_forecast = model_fit.forecast(steps=12)
future_dates = pd.date_range(start=test.index[-1], periods=12, freq='M')

# Save the results to a text file
output_path = r'C:\Users\Kasutaja\Desktop\ARIMA_Forecast_Results.txt'
with open(output_path, 'w') as f:
    f.write("ARIMA Model Summary:\n")
    f.write(str(model_fit.summary()))
    f.write("\n\nMean Squared Error (MSE):\n")
    f.write(f'{mse}\n')
    f.write("\nMean Absolute Percentage Error (MAPE):\n")
    f.write(f'{mape}%\n')
    f.write("\nFuture Forecast for the next 12 months:\n")
    for date, value in zip(future_dates, future_forecast):
        f.write(f'{date}: {value}\n')

print(f"Results have been saved to {output_path}")

# Step 9: Plot the future forecast
plt.figure(figsize=(10, 6))
plt.plot(data_cleaned['Passengers'], label='Historical Passengers')
plt.plot(future_dates, future_forecast, label='Future Forecast', color='red')
plt.title('Passenger Forecast for the Next 12 Months')
plt.legend()
plt.show()
