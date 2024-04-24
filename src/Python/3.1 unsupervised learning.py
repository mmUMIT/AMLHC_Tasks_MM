import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load in data and scale
food_data = pd.read_csv("C:/Users/m.vandermark/Desktop/studies/UMIT/Modul 12 - machine learning applications/AMLHC_Tasks_MM/src/Data/food.csv", index_col=0)
scaler = StandardScaler()
fds = scaler.fit_transform(food_data)

# Clustering with KMeans and finding the best k based on silhouette score
best_k = 0
best_silhouette = -np.inf

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(fds)
    silhouette_avg = silhouette_score(fds, cluster_labels)
    print(f"Silhouette coefficient for k = {k}: {silhouette_avg}")
    
    if silhouette_avg > best_silhouette:
        best_k = k
        best_silhouette = silhouette_avg

print(f"\nFinal selected number of clusters: {best_k}\n")

# PCA for visualization
pca = PCA(n_components=2)
data_red = pca.fit_transform(fds)
data_red *= -1  # Flip signs for consistency with R's princomp

# Plotting the results of KMeans clustering
plt.scatter(data_red[:, 0], data_red[:, 1], c=cluster_labels)
for i, txt in enumerate(food_data.index):
    plt.annotate(txt, (data_red[i, 0], data_red[i, 1]), fontsize=8)
plt.show()

# Hierarchical clustering
linked = linkage(fds, 'single')
plt.figure()
dendrogram(linked, labels=food_data.index.tolist())
plt.show()

# Heatmap (using seaborn for better visualization)
import seaborn as sns
sns.heatmap(pd.DataFrame(fds, index=food_data.index, columns=food_data.columns), annot=False)
plt.show()

# Density-based clustering with DBSCAN
dbscan = DBSCAN(eps=2, min_samples=3)
dbscan_labels = dbscan.fit_predict(fds)

# Plotting the results of DBSCAN clustering
plt.scatter(data_red[:, 0], data_red[:, 1], c=dbscan_labels)
for i, txt in enumerate(food_data.index):
    plt.annotate(txt, (data_red[i, 0], data_red[i, 1]), fontsize=8)
plt.show()
