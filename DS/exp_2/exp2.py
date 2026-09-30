import pandas as pd
import numpy as np


df = pd.read_csv("/media/galdrux/galdrux_storage/sem_5/DS/dataset/battery_features.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nNumber of Records :", df.shape[0])
print("Number of Attributes :", df.shape[1])

print("\nAttribute Names:")
print(list(df.columns))

print("\nFirst 5 Records:")
print(df.head())


attribute_mapping = {

    "cycle": {
        "category": "Numerical",
        "value_type": "Discrete",
        "scale": "Ratio",
        "justification":
            "Cycle number represents a count of battery charge/discharge "
            "cycles. It takes discrete integer values and has a meaningful "
            "zero, so it is a ratio-scale attribute."
    },

    "mean_voltage": {
        "category": "Numerical",
        "value_type": "Continuous",
        "scale": "Ratio",
        "justification":
            "Mean voltage is a measured physical quantity that can take "
            "decimal values. Voltage has a meaningful zero, so it is "
            "classified as continuous and ratio."
    },

    "min_voltage": {
        "category": "Numerical",
        "value_type": "Continuous",
        "scale": "Ratio",
        "justification":
            "Minimum voltage is a physical measurement that can take "
            "decimal values. It has a meaningful zero point."
    },

    "max_voltage": {
        "category": "Numerical",
        "value_type": "Continuous",
        "scale": "Ratio",
        "justification":
            "Maximum voltage is a continuously measured physical quantity "
            "with a meaningful zero point."
    },

    "mean_current": {
        "category": "Numerical",
        "value_type": "Continuous",
        "scale": "Ratio",
        "justification":
            "Mean current is a physical measurement that can take decimal "
            "values. Zero current represents absence of current, making "
            "the measurement ratio-scale."
    },

    "mean_temperature": {
        "category": "Numerical",
        "value_type": "Continuous",
        "scale": "Interval",
        "justification":
            "Temperature measured in degrees Celsius is continuous. "
            "Celsius has an arbitrary zero point, therefore it is an "
            "interval-scale attribute."
    },

    "max_temperature": {
        "category": "Numerical",
        "value_type": "Continuous",
        "scale": "Interval",
        "justification":
            "Maximum temperature in degrees Celsius is continuously "
            "measured. Celsius has no absolute zero, so it is interval-scale."
    },

    "discharge_time": {
        "category": "Numerical",
        "value_type": "Continuous",
        "scale": "Ratio",
        "justification":
            "Discharge time is a measured quantity that can take decimal "
            "values. Zero time represents the absence of elapsed time, "
            "therefore it is ratio-scale."
    },

    "capacity": {
        "category": "Numerical",
        "value_type": "Continuous",
        "scale": "Ratio",
        "justification":
            "Battery capacity is a measurable physical quantity that can "
            "take decimal values. It has a meaningful zero, making it "
            "continuous and ratio-scale."
    }
}


mapping_table = []

for attribute in df.columns:

    info = attribute_mapping[attribute]

    mapping_table.append({
        "Sr. No.": len(mapping_table) + 1,
        "Attribute Name": attribute,
        "Categorical / Numerical": info["category"],
        "Discrete / Continuous": info["value_type"],
        "Nominal / Ordinal / Interval / Ratio": info["scale"],
        "Justification": info["justification"]
    })


mapping_df = pd.DataFrame(mapping_table)


print("\n")
print("=" * 120)
print("ATTRIBUTE MAPPING TABLE")
print("=" * 120)

print(mapping_df.to_string(index=False))


print("\n")
print("=" * 60)
print("PYTHON DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\n")
print("=" * 60)
print("NUMBER OF UNIQUE VALUES")
print("=" * 60)

for column in df.columns:

    print(
        column,
        "->",
        df[column].nunique(),
        "unique values"
    )


print("\n")
print("=" * 60)
print("CLASSIFICATION SUMMARY")
print("=" * 60)

print(
    "\nNumerical Attributes:",
    len(
        mapping_df[
            mapping_df["Categorical / Numerical"] == "Numerical"
        ]
    )
)

print(
    "Categorical Attributes:",
    len(
        mapping_df[
            mapping_df["Categorical / Numerical"] == "Categorical"
        ]
    )
)

print(
    "Discrete Attributes:",
    len(
        mapping_df[
            mapping_df["Discrete / Continuous"] == "Discrete"
        ]
    )
)

print(
    "Continuous Attributes:",
    len(
        mapping_df[
            mapping_df["Discrete / Continuous"] == "Continuous"
        ]
    )
)


mapping_df.to_csv(
    "battery_attribute_mapping.csv",
    index=False
)

print("\nAttribute mapping table saved as:")
print("battery_attribute_mapping.csv")

print("\n")
print("=" * 60)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 60)