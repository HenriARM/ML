# import numpy as np

# Iris dataset is available in scikit
from sklearn.datasets import load_iris

x_train, y_train = load_iris(return_X_y=True)
# name of i-th flower is labels[y_train]
labels = load_iris().target_names

