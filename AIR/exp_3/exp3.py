import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
df=pd.read_csv("/media/galdrux/galdrux_storage/sem_5/AIR/dataset/battery_features.csv")
features=["mean_voltage","min_voltage","max_voltage","mean_current","mean_temperature","max_temperature","discharge_time"]
X=df[features].dropna()
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
inertia=[]
for k in range(2,9):
    model=KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(2,9),inertia,marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Squares")
plt.title("Elbow Method for K-Means")
plt.grid(True)
plt.show()
kmeans=KMeans(n_clusters=3,random_state=42,n_init=10)
clusters=kmeans.fit_predict(X_scaled)
X_result=X.copy()
X_result["Cluster"]=clusters
pca=PCA(n_components=2) 
X_pca=pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1],c=clusters,s=40)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering")
plt.grid(True)
plt.show()
cluster_counts=X_result["Cluster"].value_counts().sort_index()
print("K-Means Cluster Counts:")
print(cluster_counts)
cluster_centers=pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),columns=features)
print("\nCluster Centers:")
print(cluster_centers)
distances=np.linalg.norm(X_scaled-kmeans.cluster_centers_[clusters],axis=1)
threshold=np.percentile(distances,95)
X_result["Distance"]=distances
X_result["Anomaly"]=X_result["Distance"]>threshold
print("\nAnomaly Threshold:",threshold)
print("Number of Possible Anomalies:",X_result["Anomaly"].sum())
print("\nPossible Anomalies:")
print(X_result[X_result["Anomaly"]][features+["Cluster","Distance"]])
sample_size=min(169,len(X_scaled))
sample_indices=np.random.RandomState(42).choice(len(X_scaled),sample_size,replace=False)
X_sample=X_scaled[sample_indices]
linked=linkage(X_sample,method="ward")
plt.figure(figsize=(14,7))
dendrogram(linked,no_labels=True)
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.show()
print("\nHierarchical Clustering completed successfully.")
print("Number of samples:",len(X))
print("Number of clustering features:",len(features))
print("Number of major clusters considered:",3)