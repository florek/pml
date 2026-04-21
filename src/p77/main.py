import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X = X[:, [2, 3]]
print('Etykiety klas:', np.unique(y))
