import pickle
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class Preprocessor(TransformerMixin):
    def __init__(self, n_neighbors=5, degree=3):
        self.n_neighbors = n_neighbors
        self.degree = degree

        self.scaler = None
        self.imputer = None
        self.poly_features = None
        
    def load_model(self, filename="preprocessor_models.pkl"):
        with open(filename, 'rb') as file:
            loaded_models = pickle.load(file)

        self.scaler = loaded_models['scaler']
        self.imputer = loaded_models['imputer']
        self.poly_features = loaded_models['poly_features']

    def save_model(self, filename="preprocessor_models.pkl"):
        models = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'poly_features': self.poly_features,
        }
        with open(filename, 'wb') as file:
            pickle.dump(models, file)

    def fit(self, X, use_saved_model=False, save_model=False, filename="preprocessor_models.pkl"):
        if use_saved_model:
            self.load_model(filename)
        else:
            y = X[['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']]
            data = X.drop(columns=['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'])

            self.imputer = SimpleImputer(n_neighbors=self.n_neighbors)
            self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
            self.scaler = StandardScaler()

            imputed_data = self.imputer.fit_transform(data)
            data_poly = self.poly_features.fit_transform(imputed_data)
            self.scaler.fit(data_poly)

        if save_model:
            self.save_model(filename)

    def transform(self, X):
        data = X.drop(columns=['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'])
        if self.scaler is None or self.imputer is None or self.poly_features is None:
            raise ValueError("Preprocessor has not been fitted. Call 'fit' method first.")

        y = X[['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']]
        imputed_x = self.imputer.transform(data)
        poly_x = self.poly_features.transform(imputed_x)
        poly_columns = self.poly_features.get_feature_names_out(data.columns)
        poly_df = pd.DataFrame(poly_x, columns=poly_columns)
        X_scaled = self.scaler.transform(poly_df)

        df_result = pd.concat([y, pd.DataFrame(columns=poly_columns, data=X_scaled)], axis=1)
        return df_result
