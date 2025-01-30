from sklearn.ensemble import IsolationForest

# Sample energy consumption data (spike in usage for anomaly detection)
consumption_data = [[20], [22], [19], [23], [120], [21], [19], [25], [150], [20]]

# Train Isolation Forest model
model = IsolationForest(contamination=0.1)  # Assuming 10% anomalies
model.fit(consumption_data)

# Predict anomalies
anomalies = model.predict(consumption_data)
print("Anomalies:", anomalies)  # -1 indicates anomaly, 1 indicates normal
