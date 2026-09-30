import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv("battery_features.csv")
print(df.head())

features = [
    "mean_voltage",
    "min_voltage",
    "mean_current",
    "mean_temperature",
    "max_temperature",
    "capacity"
]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

inertia = []
sil_scores = []

for k in range(2,8):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia.append(model.inertia_)

    score = silhouette_score(
        X_scaled,
        labels
    )

    sil_scores.append(score)


    print(
        "K:",
        k,
        "Silhouette:",
        score
    )



plt.figure()

plt.plot(
    range(2,8),
    inertia,
    marker="o"
)

plt.xlabel("Number of clusters")
plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.show()

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster characteristics:")
print(
    df.groupby("cluster")[features].mean()
)

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(X_scaled)

plt.figure()

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["cluster"]
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("K-Means Battery Anomaly Clusters")

plt.show()

plt.figure()

plt.scatter(
    df["cycle"],
    df["capacity"],
    c=df["cluster"]
)

plt.xlabel("Cycle Number")
plt.ylabel("Capacity (Ah)")

plt.title("Battery Aging and Cluster Distribution")

plt.show()



