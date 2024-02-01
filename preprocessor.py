import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scaler="standard", degree=2, strategy="mean", threshold=3, remove_outliers=True, fill_na=True):
        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type, must be 'standard' or 'minmax'")
        self.remove_outliers = remove_outliers
        self.threshold = threshold
        self.fill_na = fill_na
        self.imputer = SimpleImputer(strategy=strategy)
        self.poly_features = PolynomialFeatures(degree=degree)

    def fit(self, X, y=None):
        if self.fill_na:
            X_imputed = self.imputer.fit_transform()
        else:
            X_imputed = X.copy()
        X_poly = self.poly_features.fit_transform(X_imputed)
        self.scaler.fit(X_poly)
        return self

    def transform(self, X):
        if self.fill_na:
            X_imputed = self.imputer.transform(X)
        else:
            X_imputed = X.copy()
        X_poly = self.poly_features.transform(X_imputed)
        X_scaled = self.scaler.transform(X_poly)
        if self.remove_outliers:
            # z score method to remove outliers with std greater than or less than threshold
            z_scores = np.abs((X - X.mean(axis=0))/X.std(axis=0))
            mask = (z_scores < self.threshold).all(axis=1)
            X_scaled = X_scaled[mask, :]
        return X_scaled