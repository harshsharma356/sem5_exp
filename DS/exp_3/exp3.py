import pandas as pd
import numpy as np


df = pd.read_csv("/media/galdrux/galdrux_storage/sem_5/DS/dataset/battery_features.csv")

print("=" * 70)
print("EV BATTERY DATASET")
print("=" * 70)

print(df.head())

print("\nNumber of Records:", len(df))



def calculate_mean(data):

    total = 0

    for value in data:
        total = total + value

    mean = total / len(data)

    return mean


def calculate_median(data):

    values = list(data)

    # Sort manually
    for i in range(len(values)):

        for j in range(i + 1, len(values)):

            if values[j] < values[i]:

                temp = values[i]
                values[i] = values[j]
                values[j] = temp

    n = len(values)

    if n % 2 == 1:

        median = values[n // 2]

    else:

        middle1 = values[(n // 2) - 1]
        middle2 = values[n // 2]

        median = (middle1 + middle2) / 2

    return median

def calculate_mode(data):

    values = list(data)

    frequency = {}

    for value in values:

        if value in frequency:

            frequency[value] += 1

        else:

            frequency[value] = 1

    maximum_frequency = 0

    mode = None

    for value in frequency:

        if frequency[value] > maximum_frequency:

            maximum_frequency = frequency[value]
            mode = value

    return mode, maximum_frequency

def calculate_range(data):

    minimum = data[0]
    maximum = data[0]

    for value in data:

        if value < minimum:
            minimum = value

        if value > maximum:
            maximum = value

    return maximum - minimum

def calculate_variance(data):

    mean = calculate_mean(data)

    squared_difference_sum = 0

    for value in data:

        difference = value - mean

        squared_difference_sum += difference ** 2

    # Population variance
    variance = squared_difference_sum / len(data)

    return variance

def calculate_standard_deviation(data):

    variance = calculate_variance(data)

    standard_deviation = variance ** 0.5

    return standard_deviation

def calculate_iqr(data):

    values = list(data)

    # Sort manually
    for i in range(len(values)):

        for j in range(i + 1, len(values)):

            if values[j] < values[i]:

                temp = values[i]
                values[i] = values[j]
                values[j] = temp

    n = len(values)

    # Find Q1
    if n % 2 == 0:

        lower_half = values[:n // 2]
        upper_half = values[n // 2:]

    else:

        lower_half = values[:n // 2]
        upper_half = values[(n // 2) + 1:]

    q1 = calculate_median(lower_half)
    q3 = calculate_median(upper_half)

    iqr = q3 - q1

    return q1, q3, iqr

capacity = list(df["capacity"])


print("\n")
print("=" * 70)
print("CENTRAL TENDENCY AND VARIABILITY OF BATTERY CAPACITY")
print("=" * 70)


# Mean
mean_capacity = calculate_mean(capacity)

# Median
median_capacity = calculate_median(capacity)

# Mode
mode_capacity, mode_frequency = calculate_mode(capacity)

# Range
range_capacity = calculate_range(capacity)

# Variance
variance_capacity = calculate_variance(capacity)

# Standard deviation
std_capacity = calculate_standard_deviation(capacity)

# IQR
q1_capacity, q3_capacity, iqr_capacity = calculate_iqr(capacity)


print("\nMean               :", mean_capacity)
print("Median             :", median_capacity)
print("Mode               :", mode_capacity)
print("Mode Frequency     :", mode_frequency)
print("Range              :", range_capacity)
print("Variance           :", variance_capacity)
print("Standard Deviation :", std_capacity)
print("Q1                 :", q1_capacity)
print("Q3                 :", q3_capacity)
print("Interquartile Range:", iqr_capacity)


print("\n")
print("=" * 70)
print("MULTI-ATTRIBUTE STATISTICAL ANALYSIS")
print("=" * 70)


attributes = [
    "cycle",
    "mean_voltage",
    "min_voltage",
    "max_voltage",
    "mean_current",
    "mean_temperature",
    "max_temperature",
    "discharge_time",
    "capacity"
]


results = []


for attribute in attributes:

    data = list(df[attribute])

    mean = calculate_mean(data)

    median = calculate_median(data)

    mode, frequency = calculate_mode(data)

    data_range = calculate_range(data)

    variance = calculate_variance(data)

    standard_deviation = calculate_standard_deviation(data)

    q1, q3, iqr = calculate_iqr(data)

    results.append({
        "Attribute": attribute,
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Range": data_range,
        "Variance": variance,
        "Standard Deviation": standard_deviation,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr
    })


results_df = pd.DataFrame(results)


print("\n")
print(results_df.to_string(index=False))

print("\n")
print("=" * 70)
print("GROUPED DATA ANALYSIS")
print("=" * 70)


# Divide battery cycles into groups

def cycle_group(cycle):

    if cycle <= 100:
        return "Cycles 1-100"

    elif cycle <= 200:
        return "Cycles 101-200"

    elif cycle <= 300:
        return "Cycles 201-300"

    elif cycle <= 400:
        return "Cycles 301-400"

    elif cycle <= 500:
        return "Cycles 401-500"

    else:
        return "Cycles 501+"


df["cycle_group"] = df["cycle"].apply(cycle_group)


print("\nCycle Groups:")
print(df["cycle_group"].value_counts())

print("\n")
print("=" * 70)
print("GROUPED BATTERY CAPACITY ANALYSIS")
print("=" * 70)


grouped_results = []


for group in df["cycle_group"].unique():

    group_data = list(
        df[df["cycle_group"] == group]["capacity"]
    )

    mean = calculate_mean(group_data)

    median = calculate_median(group_data)

    mode, frequency = calculate_mode(group_data)

    data_range = calculate_range(group_data)

    variance = calculate_variance(group_data)

    standard_deviation = calculate_standard_deviation(group_data)

    q1, q3, iqr = calculate_iqr(group_data)

    grouped_results.append({
        "Cycle Group": group,
        "Frequency": len(group_data),
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Range": data_range,
        "Variance": variance,
        "Standard Deviation": standard_deviation,
        "IQR": iqr
    })


grouped_df = pd.DataFrame(grouped_results)


print("\n")
print(grouped_df.to_string(index=False))


results_df.to_csv(
    "battery_statistical_analysis.csv",
    index=False
)

grouped_df.to_csv(
    "battery_grouped_analysis.csv",
    index=False
)


print("\n")
print("=" * 70)
print("RESULTS SAVED SUCCESSFULLY")
print("=" * 70)

print("1. battery_statistical_analysis.csv")
print("2. battery_grouped_analysis.csv")



print("\n")
print("=" * 70)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)