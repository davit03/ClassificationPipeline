import pickle
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class Preprocessor:
    def __init__(self, threshold=0.001, n_neighbors=5, degree=2):
        self.threshold = threshold # for future correlation analysis
        self.n_neighbors = n_neighbors
        self.degree = degree # for creating polynomial features
        
        self.scaler = None
        self.imputer = None
        self.poly_features = None

    def load_model(self, filename="preprocessor_models.pkl"):     ### HOW DO I LOAD?
        with open(filename, 'rb') as file:
            loaded_models = pickle.load(file)

        self.scaler = loaded_models['scaler']
        self.imputer = loaded_models['imputer']
        self.poly_features = loaded_models['poly_features']


    def save_model(self, filename="preprocessor_models.pkl"):     ### HOW DO I SAVE?
        models = {'scaler': self.scaler, 'imputer': self.imputer, 'poly_features': self.poly_features}
        with open(filename, 'wb') as file:
            pickle.dump(models, file)

    def fit(self, x, use_saved_model=False, save_model = False, filename = "preprocessor_models.pkl"):
        if use_saved_model:
            self.load_model(filename)
            
        else:
            self.scaler = StandardScaler()
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
            self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
            
            data = x.drop(columns=['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']) 
            
            self.imputer.fit(data) 
            imputed_data = self.imputer.fit_transform(data)
            
            self.poly_features.fit(imputed_data)
            data_poly = self.poly_features.fit_transform(imputed_data)
            
            self.scaler.fit(data_poly)
            
        if save_model:
            self.save_model(filename)

    def transform(self, X):
        data = X.drop(columns=['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']) 
        if self.scaler is None or self.imputer is None or self.poly_features is None:
            raise ValueError("Preprocessor has not been fitted. Call 'fit' method first.")
        imputed_x = self.imputer.transform(data)
        poly_x = self.poly_features.transform(imputed_x)
        poly_columns = self.poly_features.get_feature_names_out(data.columns)
        poly_df = pd.DataFrame(poly_x, columns=poly_columns)
        X_scaled = self.scaler.transform(poly_df)
        df_result = pd.DataFrame(columns = poly_columns, data = X_scaled)
        return df_result
