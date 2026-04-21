import numpy as np
import pandas as pd
from pathlib import Path


def load_std_data():
    _iris_path = Path(__file__).resolve().parent / "iris.data"
    print("Plik danych:", _iris_path)
    df = pd.read_csv(
        _iris_path,
        header=None,
        encoding="utf-8",
    )
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std, y
