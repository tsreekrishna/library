import numpy as np
from src.utils import _batch_prediction_prob
from typing import Dict, List

class model_prediction:
    '''
    model_prediction class - Base class for getting predictions from ML models
    '''
    def __init__(self,  algorithm: str, estimator) -> [List[List[int]], List[int]]:
        self.estimator = estimator
        self.algorithm = algorithm
    def fit_predict(self, X: List[List[int]], batch_size: int=8):
        if self.algorithm == 'XGB':
            pred_prob = self.estimator.predict_proba(X)
        elif self.algorithm == 'RNN':
            pred_prob = _batch_prediction_prob(X, X.shape[1], batch_size, self.estimator)
        crop_labels = np.argmax(pred_prob, axis=1)
        return pred_prob, crop_labels

class conformal_prediction(model_prediction):
    '''
    Transforms heuristic uncertainty into a rigorous form of uncertainty
    by providing prediction sets with a certain confidence as opposed to simple
    point predictions from the ML model.
    '''
    def __init__(self, algorithm, estimator):
        super().__init__(algorithm, estimator)
        self.quantile = None
        self.coverage = None
        self.true_class_scores = None
        self.n = None
    def fit(self, X_cal : List[List[int]], y_cal: List):
        cal_pred_proba = super().fit_predict(X_cal)
        true_class_prob = np.array(list(map(lambda row, idx:row[idx], cal_pred_proba, y_cal)))
        self.true_class_scores = 1 - true_class_prob
        self.n = X_cal.shape[0]
    def predict(self, X_test, alpha=0.05):
        self.coverage = (self.n+1)*(1 - alpha)/self.n
        self.quantile = np.quantile(self.true_class_scores, self.coverage)
        if self.algorithm == 'XGB':
            test_pred_proba = self.estimator.predict_proba(X_test)
        elif self.algorithm == 'RNN':
            test_pred_proba = _batch_prediction_prob(X_test, X_test.shape[1], 8, self.estimator)
        scores = 1 - test_pred_proba
        def func(crop):
            crop_set = (crop <= self.quantile).nonzero()[0]
            if len(crop_set) == 0:
                return np.nan
            else:
                return ' '.join(list(map(str, crop_set)))
        pred_set = list(map(func, scores))
        return test_pred_proba, pred_set

if __name__ == '__main__':
    pass