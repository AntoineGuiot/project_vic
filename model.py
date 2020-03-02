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
        # X = np.stack(train_set[self.predictors].values)
        if self.predictors == "hog_and_landmark":
            hog_features = np.stack(train_set['hog_features'].values)
            landmark_features = np.array([x.flatten() for x in train_set['landmarks']])
            landmark_features = landmark_features.reshape((landmark_features.shape[0], landmark_features.shape[2]))
            X = np.concatenate((hog_features, landmark_features), axis=1)

        elif self.predictors == "landmark":
            landmark_features = np.array([x.flatten() for x in train_set['landmarks']])
            landmark_features = landmark_features.reshape((landmark_features.shape[0], landmark_features.shape[2]))
            X = landmark_features

        elif self.predictors == "hog_features":
            hog_features = np.stack(train_set['hog_features'].values)
            X = hog_features

        elif self.predictors == 'lbp':
            X = np.stack(train_set['lbp'].values)

        else:
            print("Error no good predictor")
            return "Error no good predictor"
        # np.stack(data_formatter.data['hog_features'].values).shape
        print(X.shape)
        Y = train_set['emotion'].values
        self.model.fit(X, Y)

    def test(self, test_set):

        if self.predictors == "hog_and_landmark":
            hog_features = np.stack(test_set['hog_features'].values)
            landmark_features = np.array([x.flatten() for x in test_set['landmarks']])
            landmark_features = landmark_features.reshape((landmark_features.shape[0], landmark_features.shape[2]))
            X = np.concatenate((hog_features, landmark_features), axis=1)
        elif self.predictors == "landmark":
            landmark_features = np.array([x.flatten() for x in test_set['landmarks']])
            landmark_features = landmark_features.reshape((landmark_features.shape[0], landmark_features.shape[2]))
            X = landmark_features
        elif self.predictors == "hog_features":
            hog_features = np.stack(test_set['hog_features'].values)
            X = hog_features

        elif self.predictors == 'lbp':
            X = np.stack(test_set['lbp'].values)


        # X = np.stack(test_set[self.predictors].values)
        prediction = self.predict(X)
        return accuracy_score(test_set['emotion'], prediction)

    def predict(self, X):
        return self.model.predict(X)
