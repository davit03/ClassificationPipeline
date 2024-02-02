from sklearn import metrics
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC

class Model():
    def __init__(self):
        # classifiers and parameters were chosen using GridSearchCV
        estimators = [
            ('rf', RandomForestClassifier(class_weight={0: 0.5, 1: 10}, max_depth=5, random_state=21)),
            ('gbc', GradientBoostingClassifier(learning_rate=0.05, max_depth=7, random_state=21)),
            ('lgbm', LGBMClassifier(class_weight={0: 0.5, 1: 10}, num_leaves=50, max_depth=7, learning_rate=0.05, n_estimators=100, random_state=21)),
            ('svc', SVC(probability=True, C=0.01, class_weight={0: 0.5, 1: 4}, gamma=0.001, random_state=21)),
        ]

        self.model = StackingClassifier(estimators=estimators,
                                        final_estimator=SVC(probability=True, C=0.01, class_weight={0: 0.4, 1: 4}, gamma=0.001, random_state=21),
                                        cv=5)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, threshold=0.35):
        # the threshold was chosen to achieve best accuracy-sensitivity pair
        y_probs = self.model.predict_proba(X)[:, 1]
        return y_probs, threshold

    def score(self, X, y):
        pred_proba, threshold = self.predict(X)
        pred = (pred_proba > threshold).astype(int)
        fpr, tpr, _ = metrics.roc_curve(y, pred)
        tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()
        print(tn, fp, fn, tp)
        print("Accuracy: %.2f" % (100 * (tp + tn) / (tn + fp + fn + tp)))
        print("Sensitivity: %.3f" % (tp / (tp + fn)))
        print("Specificity: %.3f" % (tn / (tn + fp)))
        print("AUC: %.3f" % (metrics.auc(fpr, tpr)))
        print("MCC: %.3f" % (metrics.matthews_corrcoef(y, pred)))

    def get_params(self, deep=True):
        # to enable KFold
        return self.model