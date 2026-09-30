import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

FILE_PATH = "/media/galdrux/galdrux_storage/sem_5/DS/exp_6/Robotics_Sensor_Probability_Distribution_Dataset_2000.xlsx - Sensor_Data.csv"

df = pd.read_csv(FILE_PATH)

selected_attributes = [
    "Temperature_C",
    "Humidity_percent",
    "Battery_Voltage_V",
    "Vibration_g",
    "Motor_Speed_RPM"
]

for column in selected_attributes:
    df[column] = pd.to_numeric(df[column], errors="coerce")

data = df[selected_attributes].dropna()

print("=" * 80)
print("EXPERIMENT 6: ANALYZING AND INTERPRETING STATISTICAL DISTRIBUTIONS")
print("=" * 80)

print("\nDataset Shape:")
print(df.shape)

print("\nSelected Sensor Attributes:")
for i, column in enumerate(selected_attributes, 1):
    print(f"{i}. {column}")

print("\nNumber of records used for analysis:", len(data))

print("\nFirst 5 Records:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

print("\n" + "=" * 80)
print("1. MEAN, MEDIAN AND STANDARD DEVIATION")
print("=" * 80)

statistics_table = pd.DataFrame({
    "Mean": data.mean(),
    "Median": data.median(),
    "Standard Deviation": data.std(),
    "Minimum": data.min(),
    "Maximum": data.max()
})

print(statistics_table.round(4))

print("\n" + "=" * 80)
print("2. HISTOGRAMS")
print("=" * 80)

for column in selected_attributes:
    plt.figure(figsize=(8, 5))
    plt.hist(
        data[column],
        bins=30,
        edgecolor="black"
    )
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nHistograms generated successfully.")

print("\n" + "=" * 80)
print("3. DENSITY PLOTS")
print("=" * 80)

for column in selected_attributes:
    values = data[column].values

    kde = gaussian_kde(values)

    x = np.linspace(
        values.min(),
        values.max(),
        500
    )

    y = kde(x)

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        y,
        linewidth=2
    )

    plt.fill_between(
        x,
        y,
        alpha=0.25
    )

    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(f"Density Plot of {column}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nDensity plots generated successfully.")

print("\n" + "=" * 80)
print("4. SKEWNESS AND KURTOSIS")
print("=" * 80)

distribution_stats = pd.DataFrame({
    "Skewness": data.skew(),
    "Kurtosis": data.kurtosis()
})

print(distribution_stats.round(4))

print("\nSkewness Interpretation:")

for column in selected_attributes:

    skew = data[column].skew()

    if abs(skew) < 0.5:
        interpretation = "Approximately symmetric"
    elif 0.5 <= skew < 1:
        interpretation = "Moderately right-skewed"
    elif skew >= 1:
        interpretation = "Highly right-skewed"
    elif -1 < skew <= -0.5:
        interpretation = "Moderately left-skewed"
    else:
        interpretation = "Highly left-skewed"

    print(f"{column}: {interpretation}")

print("\nKurtosis Interpretation:")

for column in selected_attributes:

    kurtosis = data[column].kurtosis()

    if kurtosis > 0:
        interpretation = "Heavier-tailed than normal"
    elif kurtosis < 0:
        interpretation = "Lighter-tailed than normal"
    else:
        interpretation = "Similar to normal distribution"

    print(f"{column}: {interpretation}")

print("\n" + "=" * 80)
print("5. OUTLIER DETECTION USING IQR METHOD")
print("=" * 80)

outlier_results = []

for column in selected_attributes:

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ][column]

    outlier_results.append({
        "Attribute": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": len(outliers)
    })

outlier_table = pd.DataFrame(outlier_results)

print(outlier_table.round(4))

print("\n" + "=" * 80)
print("6. BOXPLOTS")
print("=" * 80)

for column in selected_attributes:

    plt.figure(figsize=(8, 4))

    plt.boxplot(
        data[column],
        vert=False
    )

    plt.xlabel(column)
    plt.title(f"Boxplot of {column}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nBoxplots generated successfully.")

print("\n" + "=" * 80)
print("7. THEORETICAL DISTRIBUTION COMPARISON")
print("=" * 80)

distribution_results = []

for column in selected_attributes:

    values = data[column].values

    normal_mu, normal_sigma = stats.norm.fit(values)

    uniform_loc, uniform_scale = stats.uniform.fit(values)

    exponential_loc, exponential_scale = stats.expon.fit(
        values,
        floc=0
    )

    distributions = {
        "Normal": (
            stats.norm,
            (normal_mu, normal_sigma)
        ),
        "Uniform": (
            stats.uniform,
            (uniform_loc, uniform_scale)
        ),
        "Exponential": (
            stats.expon,
            (exponential_loc, exponential_scale)
        )
    }

    for name, (distribution, parameters) in distributions.items():

        log_likelihood = np.sum(
            distribution.logpdf(
                values,
                *parameters
            )
        )

        number_of_parameters = len(parameters)

        AIC = (
            2 * number_of_parameters
            - 2 * log_likelihood
        )

        KS_statistic, KS_pvalue = stats.kstest(
            values,
            distribution.cdf,
            args=parameters
        )

        distribution_results.append({
            "Attribute": column,
            "Distribution": name,
            "AIC": AIC,
            "KS Statistic": KS_statistic,
            "KS p-value": KS_pvalue
        })

distribution_table = pd.DataFrame(
    distribution_results
)

print(distribution_table.round(4))

print("\nBest-fit distribution based on lowest AIC:")

best_distributions = {}

for column in selected_attributes:

    subset = distribution_table[
        distribution_table["Attribute"] == column
    ]

    best = subset.loc[
        subset["AIC"].idxmin()
    ]

    best_distributions[column] = best["Distribution"]

    print(
        f"{column}: {best['Distribution']} "
        f"(AIC = {best['AIC']:.2f})"
    )

print("\n" + "=" * 80)
print("8. OBSERVED DATA VS BEST-FIT THEORETICAL DISTRIBUTION")
print("=" * 80)

for column in selected_attributes:

    values = data[column].values

    best_distribution = best_distributions[column]

    plt.figure(figsize=(9, 5))

    plt.hist(
        values,
        bins=30,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="Observed Data"
    )

    x = np.linspace(
        values.min(),
        values.max(),
        500
    )

    if best_distribution == "Normal":

        mu, sigma = stats.norm.fit(values)

        y = stats.norm.pdf(
            x,
            mu,
            sigma
        )

    elif best_distribution == "Uniform":

        loc, scale = stats.uniform.fit(values)

        y = stats.uniform.pdf(
            x,
            loc,
            scale
        )

    elif best_distribution == "Exponential":

        loc, scale = stats.expon.fit(
            values,
            floc=0
        )

        y = stats.expon.pdf(
            x,
            loc,
            scale
        )

    plt.plot(
        x,
        y,
        linewidth=2,
        label=f"{best_distribution} Distribution"
    )

    plt.xlabel(column)
    plt.ylabel("Density")

    plt.title(
        f"{column}: Observed Data vs {best_distribution} Distribution"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nComparison plots generated successfully.")

print("\n" + "=" * 80)
print("9. Q-Q PLOTS FOR NORMALITY CHECK")
print("=" * 80)

for column in selected_attributes:

    plt.figure(figsize=(7, 5))

    stats.probplot(
        data[column],
        dist="norm",
        plot=plt
    )

    plt.title(
        f"Q-Q Plot of {column}"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nQ-Q plots generated successfully.")

print("\n" + "=" * 80)
print("10. FINAL INTERPRETATION OF EACH SENSOR VARIABLE")
print("=" * 80)

for column in selected_attributes:

    values = data[column]

    mean = values.mean()
    median = values.median()
    std = values.std()
    skew = values.skew()
    kurtosis = values.kurtosis()

    best_distribution = best_distributions[column]

    print("\n" + "-" * 80)
    print(column)
    print("-" * 80)

    print(f"Mean: {mean:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    print(f"Best-fit distribution: {best_distribution}")

    if abs(skew) < 0.5:
        print("The data is approximately symmetric.")

    elif skew > 0:
        print("The data is positively skewed.")

    else:
        print("The data is negatively skewed.")

    if kurtosis > 0:
        print(
            "The data has heavier tails and may contain "
            "more extreme observations."
        )

    elif kurtosis < 0:
        print(
            "The data has lighter tails compared with "
            "a normal distribution."
        )

    else:
        print(
            "The kurtosis is close to that of a normal distribution."
        )

print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)

print("""
The probability distributions of five selected robotic sensor
variables were analyzed using statistical and graphical techniques.

Mean, median, standard deviation, histograms, density plots,
skewness, kurtosis and IQR-based outlier detection were calculated.

The observed sensor data was also compared with Normal, Uniform
and Exponential theoretical distributions using AIC and the
Kolmogorov-Smirnov test.

These distribution characteristics help understand sensor
variability, identify abnormal readings and support statistical
analysis of robotic systems.
""")