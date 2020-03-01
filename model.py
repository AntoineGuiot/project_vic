import sklearn
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class Model:

    def __init__(self, model_type=None, hyperparameters=None, predictors=None, targets=None):
        self.model_type = model_type

        self.hyperparameters = hyperparameters
        self.predictors = predictors
        self.targets = targets

    def load(self, path):
        return

    def save(self, path):
        return

    def create_model(self):
        if self.model_type == 'SVM':
            self.model = SVC(random_state=self.hyperparameters['random_state'],
                             max_iter=self.hyperparameters['epochs'],
                             kernel=self.hyperparameters['kernel'],
                             decision_function_shape=self.hyperparameters['decision_function'],
                             gamma=self.hyperparameters['gamma'])

    def train(self, train_set):
        X = np.stack(train_set[self.predictors].values)
        # np.stack(data_formatter.data['hog_features'].values).shape
        print(X.shape)
        Y = train_set['emotion'].values
        self.model.fit(X, Y)

    def test(self, test_set):
        X = np.stack(test_set[self.predictors].values)
        prediction = self.predict(X)
        return accuracy_score(test_set['emotion'], prediction)

    def predict(self, X):
        return self.model.predict(X)
