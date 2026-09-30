import numpy as np
import pandas as pd

df = pd.read_csv("/media/galdrux/galdrux_storage/sem_5/DS/dataset/battery_features.csv")

print("\n========== ORIGINAL DATASET ==========")
print(df)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== STATISTICAL DESCRIPTION ==========")
print(df.describe())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)


print("\n========== CAPACITY COLUMN ==========")
print(df["capacity"])

print("\n========== VOLTAGE AND TEMPERATURE ==========")
print(df[["mean_voltage", "mean_temperature"]])


print("\n========== USING loc ==========")
print(df.loc[0, "capacity"])

print("\n========== USING iloc ==========")
print(df.iloc[0, 8])

print("\n========== FIRST 5 ROWS USING iloc ==========")
print(df.iloc[0:5, :])


print("\n========== CAPACITY BELOW 1.5 ==========")

low_capacity = df[df["capacity"] < 1.5]

print(low_capacity)


print("\n========== TEMPERATURE ABOVE 40°C ==========")

high_temperature = df[df["max_temperature"] > 40]

print(high_temperature)


print("\n========== MEAN VOLTAGE BELOW 3.5 V ==========")

low_voltage = df[df["mean_voltage"] < 3.5]

print(low_voltage)


print("\n========== DATA AGGREGATION ==========")

print("Average Mean Voltage:",
      df["mean_voltage"].mean())

print("Average Mean Current:",
      df["mean_current"].mean())

print("Average Temperature:",
      df["mean_temperature"].mean())

print("Average Capacity:",
      df["capacity"].mean())


print("\n========== CYCLE VALUE COUNTS ==========")

print(df["cycle"].value_counts().sort_index())


print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())


df["mean_voltage"] = df["mean_voltage"].fillna(
    df["mean_voltage"].mean()
)

df["min_voltage"] = df["min_voltage"].fillna(
    df["min_voltage"].mean()
)

df["max_voltage"] = df["max_voltage"].fillna(
    df["max_voltage"].mean()
)

df["mean_current"] = df["mean_current"].fillna(
    df["mean_current"].mean()
)

df["mean_temperature"] = df["mean_temperature"].fillna(
    df["mean_temperature"].mean()
)

df["max_temperature"] = df["max_temperature"].fillna(
    df["max_temperature"].mean()
)

df["discharge_time"] = df["discharge_time"].fillna(
    df["discharge_time"].mean()
)

df["capacity"] = df["capacity"].fillna(
    df["capacity"].mean()
)

print("\n========== AFTER HANDLING MISSING VALUES ==========")
print(df.isnull().sum())


capacity_array = np.array(df["capacity"])

voltage_array = np.array(df["mean_voltage"])

temperature_array = np.array(df["mean_temperature"])


print("\n========== CAPACITY NUMPY ARRAY ==========")
print(capacity_array)


zeros = np.zeros((3, 3))
ones = np.ones((3, 3))
sequence = np.arange(0, 20, 2)

print("\n========== ZERO ARRAY ==========")
print(zeros)

print("\n========== ONE ARRAY ==========")
print(ones)

print("\n========== ARANGE ARRAY ==========")
print(sequence)



print("\n========== CAPACITY STATISTICS ==========")

print("Mean Capacity      :", np.mean(capacity_array))
print("Median Capacity    :", np.median(capacity_array))
print("Total Capacity     :", np.sum(capacity_array))
print("Standard Deviation :", np.std(capacity_array))
print("Minimum Capacity   :", np.min(capacity_array))
print("Maximum Capacity   :", np.max(capacity_array))


cycle_array = np.array(df["cycle"])

print("\n========== UNIQUE CYCLE VALUES ==========")

print(np.unique(cycle_array))

print("\n========== NUMPY INDEXING ==========")

print("First capacity value :", capacity_array[0])
print("Second capacity value:", capacity_array[1])


print("\n========== NUMPY SLICING ==========")

print("First 10 capacity values:")

print(capacity_array[0:10])


corrected_voltage = voltage_array + 0.01

print("\n========== NUMPY BROADCASTING ==========")

print("Original Mean Voltage:")
print(voltage_array[:10])

print("\nCorrected Mean Voltage:")
print(corrected_voltage[:10])


print("\n========== NUMPY MATHEMATICAL FUNCTIONS ==========")

print("Sine of first 5 voltage values:")
print(np.sin(voltage_array[:5]))

print("\nCosine of first 5 voltage values:")
print(np.cos(voltage_array[:5]))

print("\nExponential of first 5 capacity values:")
print(np.exp(capacity_array[:5]))


print("\n========== BATTERY CAPACITY ANALYSIS ==========")

initial_capacity = df["capacity"].iloc[0]

final_capacity = df["capacity"].iloc[-1]

capacity_loss = initial_capacity - final_capacity

capacity_loss_percentage = (
    capacity_loss / initial_capacity
) * 100


print("Initial Capacity       :", initial_capacity)
print("Final Capacity         :", final_capacity)
print("Capacity Loss          :", capacity_loss)
print("Capacity Loss (%)      :", capacity_loss_percentage)

print("\n========== TEMPERATURE ANALYSIS ==========")

print("Average Temperature :",
      df["mean_temperature"].mean())

print("Maximum Temperature :",
      df["max_temperature"].max())


print("\n========== DISCHARGE TIME ANALYSIS ==========")

print("Average Discharge Time :",
      df["discharge_time"].mean())

print("Minimum Discharge Time :",
      df["discharge_time"].min())

print("Maximum Discharge Time :",
      df["discharge_time"].max())

df.to_csv(
    "processed_battery_data.csv",
    index=False
)

print("\nProcessed dataset saved successfully.")


print("\n========== EXPERIMENT COMPLETED ==========")

print("Battery dataset successfully analyzed")
print("using Pandas and NumPy.")