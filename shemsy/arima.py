from statsmodels.tsa.arima.model import ARIMA

# Assuming 'energy_usage' column stores energy consumption data
model = ARIMA(data['energy_usage'], order=(5, 1, 0))  # ARIMA parameters (p, d, q)
model_fit = model.fit()

# Predict future energy usage
forecast = model_fit.forecast(steps=24)  # Forecasting for next 24 hours
print("Energy Forecast for next 24 hours:", forecast)
