from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scaler="standard", degree=2, strategy="mean", fill_na=True):
        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            # to deal with the problem with make_pipeline
            self.scaler = scaler
        self.fill_na = fill_na
        self.degree = degree
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.poly_features = PolynomialFeatures(degree=self.degree)

    def fit(self, X, y=None):
        if self.fill_na:
            X_imputed = self.imputer.fit_transform(X)
            X_poly = self.poly_features.fit_transform(X_imputed)
        else:
            X_poly = X.copy()
        self.scaler.fit(X_poly)
        return self

    def transform(self, X):
        if self.fill_na:
            X_imputed = self.imputer.transform(X)
            X_poly = self.poly_features.transform(X_imputed)
        else:
            X_poly = X.copy()
        X_scaled = self.scaler.transform(X_poly)
        return X_scaled