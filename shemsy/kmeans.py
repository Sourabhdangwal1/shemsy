from sklearn.cluster import KMeans

# Example device usage data (replace with real data)
device_usage = np.array([[10], [20], [30], [15], [25], [35]])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(device_usage)

# Display cluster centers and device labels
print("Cluster Centers:", kmeans.cluster_centers_)
print("Device Usage Labels:", kmeans.labels_)
