import argparse
import pickle
import json
import pandas as pd
from model import Model
from preprocessor import Preprocessor

class Pipeline:
    def __init__(self, test_mode=False, save_models=False):
        self.test_mode = test_mode
        self.save_models = save_models
        self.model = Model()
        self.preprocessor = Preprocessor(scaler='minmax')

    def run(self, datapath):
        dataset = pd.read_csv(datapath)

        if self.test_mode:
            with open('preprocessor.pkl', 'rb') as file:
                loaded_preprocessor = pickle.load(file)
            self.preprocessor = loaded_preprocessor
            with open('model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
            self.model = loaded_model

            X = self.preprocessor.transform(dataset)
            results = self.model.predict(X)
            dictionary = {'predict_probas': results[0].tolist(), 'threshold': results[1]}
            with open("predictions.json", "w") as outfile: 
                json.dump(dictionary, outfile)
            
        else:
            targets = dataset['In-hospital_death']
            features = dataset.drop(columns=['In-hospital_death'])

            X = self.preprocessor.fit_transform(features)
            self.model.fit(X, targets)
            if self.save_models:
                pickle.dump(self.model, open('model.pkl', 'wb'))
                pickle.dump(self.preprocessor, open('preprocessor.pkl', 'wb'))
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run in test mode if --test is provided, provide the path to the dataset using --data_path, use --save_models to save the trained models")
    parser.add_argument("--test", action="store_true", help="Run in test mode.")
    parser.add_argument("--data_path", help="Access data for training or test")
    parser.add_argument("--save_models", action="store_true", help="Save trained model")

    args = parser.parse_args()
    
    pipeline = Pipeline(test_mode=args.test, save_models=args.save_models)
    pipeline.run(datapath=args.data_path)
