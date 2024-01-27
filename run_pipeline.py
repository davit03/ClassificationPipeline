import argparse
import pandas as pd
from model import Model
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self,test_mode = False):
        self.test_mode = test_mode
        self.model = Model()
        self.preprocessor = Preprocessor()

    def run(self, datapath):
        dataset = pd.read_csv(datapath)
        
        if self.test_mode:
            self.preprocessor.fit(dataset, use_saved_model=True)
            self.preprocessor.transform(dataset)
            # continue for the model part
        else:
            targets = dataset['In-hospital_death']
            features = dataset.drop(columns=['In-hospital_death'])
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
            self.preprocessor.fit(X_train)
            self.preprocessor.transform(X_train)
            # continue for the model part
              
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run in test mode if --test is provided.")
    parser.add_argument("--test", action="store_true", help="Run in test mode.")
    parser.add_argument("--data_path", help="Access data for training or test")
    args = parser.parse_args()
    
    pipeline = Pipeline(test_mode=args.test)
    pipeline.run(datapath= args.data_path)
