import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from .metrics import dist_with_miss


class KNNMVImputer (BaseEstimator, TransformerMixin):
    """Imputation transformer for completing missing values.

    Parameters
    ----------
    strategy : string, optional (default="mean")
        The imputation strategy.
        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
    k : integer, optional (default=3)
    l : float, optional (default=0.0)
    dist : function, optional (default=knnmv.metrics.dist_with_miss)
    scaler : TransformerMixin, optional
                (default=sklearn.preprocessing.MinMaxScaler)

    Examples
    --------
    >>> import numpy as np
    >>> from knnmv.impute import KNNMVImputer
    >>> import sklearn.datasets
    >>> X,y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    >>> miss_perc = 25
    >>> msk = np.random.rand(np.shape(X)[0],np.shape(X)[1]) <= miss_perc/ 100
    >>> Xm = X.copy()
    >>> Xm[msk] = np.nan
    >>> knnmv_imp = KNNMVImputer()
    >>> Xm_i = knnmv_imp.fit_transform(Xm)

    Notes
    -----
    Missing values are imputed using the features’ median value of the K
    closest samples, and in the very specific case not to find at least
    one non-missing value in the K retrieved neighbours, the median of
    the whole column is used.
    """

    def __init__(self, missing_values=np.nan, strategy="median", k=3, l=0.0,
                 dist=dist_with_miss, scaler=MinMaxScaler()):
        self.missing_values = missing_values
        self.strategy = strategy
        self.k = k
        self.l = l
        self.dist = dist
        self.scaler = scaler
        self.statistics_ = None

    def _validate_input(self, X):
        allowed_strategies = ["mean", "median"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("KNNMV does not support data with dtype "
                             "{0}. Please provide either a numeric array (with"
                             " a floating point or integer dtype) or "
                             "categorical data represented either as an array "
                             "with integer dtype or an array of string values "
                             "with an object dtype.".format(X.dtype))

        return X

    def fit(self, X, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : KNNMVImputer
        """

        X = self._validate_input(X)

        self.statistics_ = np.shape(X)[1]  # m - number of features
        self.scaler.fit(X)
        self.Xt_d = self.scaler.transform(X)

        return self

    def transform(self, X):
        """Impute all missing values in X.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        ----------
        X_new : {array-like}, shape (n_samples, n_features)
            Transformed array.
        """

        if (self.statistics_ is None):
            return X

        if (X.shape[1] != self.statistics_):
            raise ValueError("X has %d features per sample, expected %d"
                             % (X.shape[1], self.statistics_))

        X = self._validate_input(X)
        Xt_in = self.scaler.transform(X)

        for i in range(len(Xt_in)):
            row = Xt_in[i, :]
            if(np.sum(np.isnan(row)) > 0):

                # Retrieve the KNN for row i
                # 1) Compute the distance for all the rows
                dists = np.inf * np.ones(len(self.Xt_d))
                for j in range(len(self.Xt_d)):
                    dists[j] = self.dist(row, self.Xt_d[j, :], l=self.l)
                knn = np.argsort(dists)[1:self.k+1]
                nanidxs = np.where(np.isnan(row) == True)[0]
                for nanidx in nanidxs:
                    vls = self.Xt_d[knn, nanidx]
                    vls = vls[~np.isnan(vls)]
                    if(len(vls) == 0):
                        # all the nearest neighbors are NaNs
                        # take the median of the whole column
                        all_vls = self.Xt_d[:, nanidx]
                        all_vls = all_vls[~np.isnan(all_vls)]
                        md = np.median(all_vls)
                    else:
                        md = np.median(vls)
                    Xt_in[i, nanidx] = md

        return Xt_in

    def fit_transform(self, X, y=None):
        """Fit the imputer on X and impute all missing values in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        Returns
        ----------
        X_new : {array-like}, shape (n_samples, n_features)
            Transformed array.
        """
        return self.fit(X).transform(X)
