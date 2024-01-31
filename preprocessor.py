import pickle
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler, VarianceThreshold, SelectKBest

class Preprocessor(TransformerMixin):
    def __init__(self, threshold=0.001, n_neighbors=5, degree=2, k_best=10):
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.degree = degree
        self.k_best = k_best

        self.scaler = None
        self.imputer = None
        self.poly_features = None
        self.min_max_scaler = None
        self.robust_scaler = None
        self.variance_selector = None
        self.k_best_selector = None

    def load_model(self, filename="preprocessor_models.pkl"):
        with open(filename, 'rb') as file:
            loaded_models = pickle.load(file)

        self.scaler = loaded_models['scaler']
        self.imputer = loaded_models['imputer']
        self.poly_features = loaded_models['poly_features']
        self.min_max_scaler = loaded_models.get('min_max_scaler')
        self.robust_scaler = loaded_models.get('robust_scaler')
        self.variance_selector = loaded_models.get('variance_selector')
        self.k_best_selector = loaded_models.get('k_best_selector')

    def save_model(self, filename="preprocessor_models.pkl"):
        models = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'poly_features': self.poly_features,
            'min_max_scaler': self.min_max_scaler,
            'robust_scaler': self.robust_scaler,
            'variance_selector': self.variance_selector,
            'k_best_selector': self.k_best_selector
        }
        with open(filename, 'wb') as file:
            pickle.dump(models, file)

    def fit(self, X, use_saved_model=False, save_model=False, filename="preprocessor_models.pkl"):
        if use_saved_model:
            self.load_model(filename)
        else:
            y = X[['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']]  # Separate target variable
            data = X.drop(columns=['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'])

            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
            self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
            self.scaler = StandardScaler()
            self.min_max_scaler = MinMaxScaler()
            self.robust_scaler = RobustScaler()
            self.variance_selector = VarianceThreshold(threshold=self.threshold)
            self.k_best_selector = SelectKBest(k=self.k_best)

            imputed_data = self.imputer.fit_transform(data)
            data_poly = self.poly_features.fit_transform(imputed_data)
            data_poly_scaled = self.scaler.fit_transform(data_poly)
            data_poly_min_max_scaled = self.min_max_scaler.fit_transform(data_poly_scaled)
            data_poly_robust_scaled = self.robust_scaler.fit_transform(data_poly_min_max_scaled)
            data_poly_variance_selected = self.variance_selector.fit_transform(data_poly_robust_scaled)
            data_poly_k_best_selected = self.k_best_selector.fit_transform(data_poly_variance_selected)

        if save_model:
            self.save_model(filename)

    def transform(self, X):
        data = X.drop(columns=['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'])
        if self.scaler is None or self.imputer is None or self.poly_features is None:
            raise ValueError("Preprocessor has not been fitted. Call 'fit' method first.")

        y = X[['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']]
        # Transform features
        imputed_x = self.imputer.transform(data)
        poly_x = self.poly_features.transform(imputed_x)
        poly_columns = self.poly_features.get_feature_names_out(data.columns)
        poly_df = pd.DataFrame(poly_x, columns=poly_columns)
        X_scaled = self.scaler.transform(poly_df)

        feature_transformers = [
            ('min_max_scaler', self.min_max_scaler),
            ('robust_scaler', self.robust_scaler),
            ('variance_selector', self.variance_selector),
            ('k_best_selector', self.k_best_selector),
            ('poly_features', self.poly_features),
            ('standard_scaler', self.scaler),
            ('knn_imputer', self.imputer)
        ]

        additional_dfs = [transformer.transform(X_scaled) for _, transformer in feature_transformers if transformer]

        if additional_dfs:
            X_scaled = pd.concat([X_scaled] + additional_dfs, axis=1)

        df_result = pd.concat([y, pd.DataFrame(columns=poly_columns, data=X_scaled)], axis=1)
        return df_result
