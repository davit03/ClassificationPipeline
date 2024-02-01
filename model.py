from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from preprocessor import Preprocessor
class Model():
    # class_weights were chosen based on compute_class_weight function and also GridSearchCV
    def __init__(self):
        estimators = [
            ('nb', make_pipeline(Preprocessor(scaler="minmax", degree=1),
                                 ComplementNB(alpha=0.0001))),
            ('lr', make_pipeline(Preprocessor(),
                                 LogisticRegression(max_iter=1000, class_weight="balanced"))),
            ('dt', DecisionTreeClassifier(class_weight={0: 0.5, 1: 3.5}, random_state=21)),
            ('lda', make_pipeline(Preprocessor(degree=1),LinearDiscriminantAnalysis())),
        ]

        self.model = StackingClassifier(estimators=estimators,
                                        final_estimator=RandomForestClassifier(class_weight={0: 3.7, 1: 0.3}, random_state=21),
                                        cv=5)
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_probs = self.model.predict_proba(X)[:,1]
        return y_probs, 0.5
