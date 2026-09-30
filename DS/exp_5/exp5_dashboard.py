import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_FILE = "/media/galdrux/galdrux_storage/sem_5/DS/dataset/battery_features.csv"

df = pd.read_csv(CSV_FILE)

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

target = "capacity"

X = df.drop(columns=[target])
y = df[target]

print("\nIndependent Variables:")
print(X.columns.tolist())

print("\nDependent Variable:")
print(target)

print("\nDataset Description:")
print(df.describe())

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

correlation = df.corr(numeric_only=True)[target].drop(target)
correlation = correlation.sort_values(key=abs, ascending=False)

print("\nCorrelation with Capacity:")
print(correlation)

plt.figure(figsize=(10, 5))
correlation.plot(kind="bar")
plt.title("Feature Correlation with Capacity")
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

filter_selector = SelectKBest(score_func=f_regression, k=4)
filter_selector.fit(X_train, y_train)

filter_features = X_train.columns[filter_selector.get_support()].tolist()

filter_scores = pd.Series(
    filter_selector.scores_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nFilter Method - F Scores:")
print(filter_scores)

print("\nSelected Features using Filter Method:")
print(filter_features)

plt.figure(figsize=(10, 5))
filter_scores.plot(kind="bar")
plt.title("Filter Method - F Scores")
plt.xlabel("Features")
plt.ylabel("F Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

rfe_model = LinearRegression()

rfe = RFE(
    estimator=rfe_model,
    n_features_to_select=4
)

rfe.fit(X_train, y_train)

rfe_features = X_train.columns[rfe.support_].tolist()

rfe_ranking = pd.Series(
    rfe.ranking_,
    index=X_train.columns
).sort_values()

print("\nRFE Feature Ranking:")
print(rfe_ranking)

print("\nSelected Features using RFE:")
print(rfe_features)

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)

tree_importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

tree_features = tree_importance.head(4).index.tolist()

print("\nRandom Forest Feature Importance:")
print(tree_importance)

print("\nSelected Features using Tree-Based Method:")
print(tree_features)

plt.figure(figsize=(10, 5))
tree_importance.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nFeature Selection Comparison")
print("--------------------------------")

print("Filter Method:", filter_features)
print("RFE Method:", rfe_features)
print("Tree Method:", tree_features)

selected_features = {
    "All Features": X.columns.tolist(),
    "Filter Method": filter_features,
    "RFE Method": rfe_features,
    "Tree Method": tree_features
}

results = []

for method, features in selected_features.items():

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    start_time = time.perf_counter()

    model.fit(X_train[features], y_train)

    train_time = time.perf_counter() - start_time

    predictions = model.predict(X_test[features])

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Method": method,
        "Number of Features": len(features),
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "Training Time (s)": train_time
    })

results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.bar(results_df["Method"], results_df["R2 Score"])
plt.title("R2 Score Comparison")
plt.xlabel("Feature Selection Method")
plt.ylabel("R2 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(results_df["Method"], results_df["RMSE"])
plt.title("RMSE Comparison")
plt.xlabel("Feature Selection Method")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

print("\nFinal Selected Features")
print("-----------------------")
print("Filter Method:", filter_features)
print("RFE Method:", rfe_features)
print("Tree-Based Method:", tree_features)

common_features = set(filter_features) & set(rfe_features) & set(tree_features)

print("\nCommon Features Selected by All Three Methods:")
print(list(common_features))

print("\nExperiment Completed Successfully.")