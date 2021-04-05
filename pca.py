from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

np.random.seed(10)
X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.0)

# Kernel PCA
gamma = 0.2
# TODO: we leaened PCA from calculating PC1, macx point distance on linear projection, whats this?
Cov = np.exp(-gamma * pdist(X, metric='euclidean'))
Cov = squareform(Cov)  # reshape from flat to square

# TODO:?
lam, v = np.linalg.eig(Cov)
idxes = lam.argsort()
lam = lam[idxes]
v = v[:, idxes]
plt.scatter(v[:, -1], v[:, -2], c=y)
plt.show()



# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
# import numpy as np
# import sklearn.decomposition
#
# np.random.seed(10)
# X, y = make_blobs(n_samples=1000, n_features=4, centers=4, cluster_std=3.0)
#
# # pca = sklearn.decomposition.PCA(n_components=2, whiten=True)
# # pca = sklearn.decomposition.KernelPCA(n_components=2, gamma=0.1)
# # pca = sklearn.decomposition.SparsePCA(n_components=2)
# pca = sklearn.decomposition.IncrementalPCA(n_components=2)
# # pca.fit()
# # pca.transform()
# X_p = pca.fit_transform(X)
#
# plt.scatter(X_p[:, -1], X_p[:, -2], c=y)
# plt.show()
