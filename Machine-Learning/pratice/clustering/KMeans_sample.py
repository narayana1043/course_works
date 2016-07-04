# sample type of clustering

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 1.06],
              [9, 11]])

# plt.scatter(X[:,0], X[:,1])
# plt.show()

clf = KMeans(n_clusters=6)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = 10 * ["g.", "r.", "c.", "b.", "k."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=20)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.show()
