import argparse
import pickle
import pandas as pd
from model import Model
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self,test_mode = False, save_models = False):
        self.test_mode = test_mode
        self.save_models = save_models
        self.model = Model()
        self.preprocessor = Preprocessor()

    def run(self, datapath):
        dataset = pd.read_csv(datapath)
        targets = dataset['In-hospital_death']
        features = dataset.drop(columns=['In-hospital_death'])

        # if test mode, previously saved models are used 
        if self.test_mode:
            # loading trained model
            with open('model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
            self.model = loaded_model
            # loading preprocessor model
            with open('preprocess_model.pkl', 'rb') as f:
                prep_model = pickle.load(f)
            self.preprocessor = prep_model
            
            preprocessed_features = self.preprocessor.transform(features)
            self.model.predict(preprocessed_features)
        
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
            
            self.preprocessor.fit(X_train)
            preprocessed_features = self.preprocessor.transform(X_train)
            preprocessed_test_features = self.preprocessor.transform(X_test)
            
            self.model.fit(preprocessed_features, y_train)
            self.model.predict(preprocessed_test_features)  
            
            if self.save_models:
                pickle.dump(self.preprocessor, open('preprocess_model.pkl', 'wb'))                
                pickle.dump(self.model, open('model.pkl', 'wb'))
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run in test mode if --test is provided, provide the path to the dataset using --data_path, use --save_models to save preprocessor and the trained models")
    parser.add_argument("--test", action="store_true", help="Run in test mode.")
    parser.add_argument("--data_path", help="Access data for training or test")
    parser.add_argument("--save_models", action="store_true", help="Save preprocessor and trained model")

    args = parser.parse_args()
    
    pipeline = Pipeline(test_mode=args.test, save_models = args.save_models)
    pipeline.run(datapath= args.data_path)
