# pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
import numpy as np

breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features.values.astype(float)
y = breast_cancer.data.targets.values.flatten()

y = np.where(y == 'M', 1, 2)
X = (X - X.mean(axis=0)) / X.std(axis=0)

data_clean = np.column_stack([y, X])
np.savetxt("wdbc_clean.txt", data_clean, fmt="%.7e")

print(data_clean.shape)