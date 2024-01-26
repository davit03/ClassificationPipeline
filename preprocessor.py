import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class Preprocessor:
    def __init__(self, threshold=0.001, n_neighbors=5, degree=2):
        self.threshold = threshold # for future correlation analysis
        self.n_neighbors = n_neighbors
        self.degree = degree # for creating polynomial features
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.scaler = StandardScaler()

    def fit(self, X_train):
        X_data = X_train.drop(columns=['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']) 
        # filled nan using knn method
        X_imputed = self.knn_imputer.fit_transform(X_data) 
        # finding the features with lowest correlation
        correlation_matrix = X_imputed.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
        low_correlation_features = [column for column in upper_triangle.columns if any(upper_triangle[column] < self.threshold)]
        selected = X_imputed[low_correlation_features]
        # creating polynomial features
        X_poly = self.poly_features.fit_transform(selected)
        poly_feature_names = self.poly_features.get_feature_names(low_correlation_features)
        df_poly = pd.DataFrame(X_poly, columns=poly_feature_names)
        X = pd.concat([X_imputed, df_poly], axis=1)
        # scaling        
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)

    def transform(self, X_test):
        X_imputed = self.knn_imputer.transform(X_test)
        X_poly = self.poly_features.transform(X_imputed)
        X_scaled = self.scaler.transform(X_poly)
        df_result = pd.DataFrame(X_scaled, columns=self.columns_after_processing)
        return df_result
