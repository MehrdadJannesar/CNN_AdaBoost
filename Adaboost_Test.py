import numpy as np
import copy

from decisionstump import DecisionStumpClassifier

class AdaBoostClassifier:

    def __init__(self, n_classes_,
                 weak_classifier_ = DecisionStumpClassifier()):
        self.mxWC = n_classes_
        self.WCClass = weak_classifier_


    def train(self, X_, y_, verbose=False):


        X = np.array(X_)
        y = np.array(y_).flatten(1)
        y[y == 0] = -1
        n_samples, n_features = X.shape

        assert n_samples == y.size
        
        # Initialize weak classifiers
        self.WCs = [copy.deepcopy(self.WCClass) for i in range(self.mxWC)]
        self.nWC = 0
        self.alpha = np.zeros((self.mxWC))
        self.features = n_features
        self.sum_eval = 0

        # Initialize weights of inputs samples
        W = np.ones((n_samples)) / n_samples

        for i in range(self.mxWC):
            if verbose: print('Training %d-th weak classifier' % i)
            err = self.WCs[i].train(X, y, W)
            h = self.WCs[i].predict(X).flatten(1)
            self.alpha[i] = 0.5 * np.log((1 - err) / err)
            W = W * np.exp(-self.alpha[i]*y*h)
            W = W / W.sum()
            self.nWC = i+1
            if verbose: print('%d-th weak classifier: err = %f' % (i, err))
            if self._evaluate(i, h, y) == 0:
                print(self.nWC, "weak classifiers are enought to make error rate reach 0.0")
                break

    def _evaluate(self, t, h, y):


        self.sum_eval = self.sum_eval + h*self.alpha[t]
        yPred = np.sign(self.sum_eval)
        return np.sum(yPred != y)

    def predict(self, test_set_):


        hsum = self.weightedSum(test_set_)
        CI = abs(hsum) / np.sum(abs(self.alpha))

        yPred = np.sign(hsum)
        yPred[yPred == -1] = 0

        return yPred, CI

    def weightedSum(self, test_set_):


        test_set = np.array(test_set_)

        assert test_set.shape[1] == self.features

        hsum = 0
        for i in range(self.nWC):
            hsum = hsum + self.alpha[i] * self.WCs[i].predict(test_set)

        return hsum