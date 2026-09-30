import scipy.io
import numpy as np
import pandas as pd


# ==============================
# Load NASA Battery Dataset
# ==============================

mat = scipy.io.loadmat("/media/galdrux/galdrux_storage/sem_5/AIR/ai_use_case/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat")

battery = mat["B0005"][0, 0]

cycles = battery["cycle"][0]


# ==============================
# Extract discharge cycles
# ==============================

dataset = []


for i, c in enumerate(cycles):

    # c is already a MATLAB struct
    cycle_type = c["type"][0]

    if cycle_type == "discharge":

        data = c["data"][0, 0]


        voltage = data["Voltage_measured"][0]
        current = data["Current_measured"][0]
        temperature = data["Temperature_measured"][0]
        time = data["Time"][0]

        capacity = data["Capacity"][0][0]


        features = {

            "cycle": i,

            "mean_voltage": np.mean(voltage),

            "min_voltage": np.min(voltage),

            "max_voltage": np.max(voltage),

            "mean_current": np.mean(current),

            "mean_temperature": np.mean(temperature),

            "max_temperature": np.max(temperature),

            "discharge_time": time[-1],

            "capacity": capacity

        }


        dataset.append(features)



# ==============================
# Convert to DataFrame
# ==============================

df = pd.DataFrame(dataset)


print("\nExtracted discharge cycles:")
print(df.head())


print("\nDataset shape:")
print(df.shape)


# Save processed data

df.to_csv(
    "battery_features.csv",
    index=False
)


print("\nSaved: battery_features.csv")