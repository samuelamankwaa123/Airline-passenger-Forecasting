# Airline-passenger-Forecasting
Time Series Forecasting with ARIMA
This project uses a time series analysis and forecasting approach to predict international airline passengers over a future period based on historical data. The primary method employed is the ARIMA model (AutoRegressive Integrated Moving Average), which is suitable for time series data with trends and seasonality.

Project Overview
This repository contains:

The cleaned airline passenger dataset.
Python script that demonstrates the following steps:
Data cleaning and preprocessing.
Visualization of the time series.
Decomposition of the time series into trend, seasonality, and residual components.
Implementation of an ARIMA model to forecast future values.
Evaluation of the model using performance metrics like Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE).
Visualization of predicted values alongside actual historical data.
Dataset
The dataset contains the monthly number of international airline passengers from January 1949 to December 1969. The data is available as a CSV file and consists of the following columns:

Month: The month in the format YYYY-MM.
Passengers: The number of passengers (in thousands).
The dataset was pre-processed to handle missing values and correct formatting issues before modeling.

Steps Followed
1. Data Preprocessing
Loaded and cleaned the dataset by ensuring the Month column was correctly formatted as a date.
Handled any missing or incorrectly formatted rows.
2. Data Visualization
Plotted the time series data to visually inspect trends, seasonality, and overall patterns.
3. Time Series Decomposition
Decomposed the time series into its trend, seasonality, and residuals using seasonal decomposition.
4. ARIMA Model
Implemented the ARIMA model with parameters (5, 1, 0), which were chosen based on an initial inspection of autocorrelations and differencing.
The model was trained on 80% of the data, with the remaining 20% used for testing.
5. Model Evaluation
Mean Squared Error (MSE): The model achieved an MSE of 6499.09 on the test data.
Mean Absolute Percentage Error (MAPE): The model achieved a MAPE of 16.14%, which indicates that the predictions are on average 16% off from the actual values.
6. Forecasting Future Values
Forecasted passenger numbers for the next 12 months, showing the predicted passenger count alongside the historical data.
Results
The model was able to capture the upward trend and seasonal pattern in the data.
The future forecast suggests a slight decrease in passenger growth, with values stabilizing around 460-480 passengers per month.
Visualizations
Time Series Decomposition: Displays the trend, seasonality, and residual components of the time series.
Actual vs Predicted: Shows the comparison between actual historical data and ARIMA predictions.
Future Forecast: Visualizes the 12-month forecast alongside historical data.
Files
international-airline-passengers.csv: The dataset used for analysis.
ARIMA_Forecast_Results.txt: The output of the model, including forecasted values and evaluation metrics.
Time_series_analysis.py: Python code containing the full analysis.
Plots and figures showing time series decomposition and forecasts.
Usage
