# KNNMV - Sparsity Aware K - Nearest Neighbours

This is a Python implementation of a KNN imputer that specifically takes into account missing values in the distance metric.
It has shown to outperform the default XGBoost built-in strategy and a simple column-wise median imputation. <br />
Further infos about the theoretical background of this research can be found in the related post on __Medium__.

## Installation
```sh
pip install knnmv
```

## Usage
```python
import numpy as np
from knnmv.impute import KNNMVImputer
import sklearn.datasets
X,y = sklearn.datasets.load_breast_cancer(return_X_y=True)
miss_perc = 25
msk = np.random.rand(np.shape(X)[0],np.shape(X)[1]) <= miss_perc/ 100
Xm = X.copy()
Xm[msk] = np.nan
knnmv_imp = KNNMVImputer(strategy="median", k=5, l=0.25)
Xm_i = knnmv_imp.fit_transform(Xm)
```


