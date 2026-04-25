!pip install scikit-learn pandas matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("Cluster Counts:")
print(df['Cluster'].value_counts())
print("\nCluster Means:")
print(df.groupby('Cluster').mean())
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KMeans Clustering (Wine Dataset)")
plt.show()
