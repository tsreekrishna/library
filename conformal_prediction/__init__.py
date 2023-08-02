class conformal_prediction:
    def __init__(self, estimator):
        self.estimator = estimator
        self.quantile = None
        self.coverage = None
    def fit(self, X_cal, y_cal, alpha):
        cal_pred_proba = self.estimator.predict_proba(X_cal)
        scores = 1 - cal_pred_proba
        true_class_scores = list(map(lambda row, idx:row[idx], scores, y_cal))
        n = X_cal.shape[0]
        self.coverage = (n+1)*(1 - alpha)/n
        self.quantile = np.quantile(true_class_scores, self.coverage)
    def predict(self, X_test):
        test_pred_proba = self.estimator.predict_proba(X_test)
        scores = 1 - test_pred_proba
        def func(crop):
            crop_set = (crop <= self.quantile).nonzero()[0]
            if len(crop_set) == 0:
                return np.nan
            else:
                return ' '.join(list(map(str, crop_set)))
        pred_set = list(map(func, scores))
        return test_pred_proba, pred_set