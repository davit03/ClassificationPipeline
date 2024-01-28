from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor
from sklearn import metrics
class Model():
    def __init__(self):
        self.model = KNeighborsClassifier()
        self.model = LogisticRegression()
        self.model = GaussianNB()
        self.model = LinearDiscriminantAnalysis()
        self.model = QuadraticDiscriminantAnalysis()
        self.model = SVC(C=1e8, kernel='poly', probability=True)
        self.model = DecisionTreeClassifier()
        self.model = RandomForestClassifier()
        self.model = BaggingClassifier()
        self.model = GradientBoostingClassifier()
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        pass
    def score(self, X, y):
        pred = self.model.predict(X)
        fpr, tpr, _ = metrics.roc_curve(y, pred)
        tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()
        print("Accuracy: %.2f" % (100 * (tp + tn) / (tn + fp + fn + tp)))
        print("Sensitivity: %.3f" % (tp / (tp + fn)))
        print("Specificity: %.3f" % (tn / (tn + fp)))
        print("AUC: %.3f" % (metrics.auc(fpr, tpr)))
        print("MCC: %.3f" % (metrics.matthews_corrcoef(y, pred)))

data = pd.read_csv("Survival_dataset.csv")
X = data.drop("In-hospital_death", axis = 1)
y = data["In-hospital_death"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
preprocessor = Preprocessor()
preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)
model = Model()
model.fit(X_train, y_train)
model.score(X_test, y_test)
