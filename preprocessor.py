from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scaler="standard", degree=2, strategy="mean", threshold=0.1, fill_na=True, poly=False):
        self.fill_na = fill_na
        self.degree = degree
        self.strategy = strategy
        self.threshold = threshold
        self.var_thresh = VarianceThreshold(threshold=self.threshold)
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.poly = poly
        if self.poly:
            self.poly_features = PolynomialFeatures(degree=self.degree)
        else:
            self.poly_features = PolynomialFeatures(degree=1)
        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            # to deal with the problem with make_pipeline
            self.scaler = scaler

    def fit(self, X, y=None):
        X_reduced = self.var_thresh.fit_transform(X)
        if self.fill_na:
            X_imputed = self.imputer.fit_transform(X_reduced)
            X_poly = self.poly_features.fit_transform(X_imputed)
        else:
            X_poly = X_reduced
        self.scaler.fit(X_poly)
        return self

    def transform(self, X):
        X_reduced = self.var_thresh.transform(X)
        if self.fill_na:
            X_imputed = self.imputer.transform(X_reduced)
            X_poly = self.poly_features.transform(X_imputed)
        else:
            X_poly = X_reduced
        X_scaled = self.scaler.transform(X_poly)
        return X_scaled