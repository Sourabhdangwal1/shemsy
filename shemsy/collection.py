import pandas as pd
import numpy as np

# Sample energy consumption data (usually this would come from smart devices or IoT sensors)
data = pd.read_csv('energy_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Inspecting the data
print(data.head())
