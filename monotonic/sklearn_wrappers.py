from sklearn.base import BaseEstimator, ClassifierMixin
import monotonic.monotonic.utils as monotonic_utils
import numpy as np
import pandas as pd
import python_utils.python_utils.caching as caching

class monotonic_predictor(ClassifierMixin):

    def __init__(self, horse):
        self.horse = horse

    def decision_function(self, X):
        test_data = monotonic_utils.data(hash(caching.get_hash(X)), range(len(X)), X, [None for i in xrange(len(X))])
        return np.array([self.horse(datum) for datum in test_data])

    def train_info(self):
        return self.horse.train_info()
        
        
class monotonic_fitter(BaseEstimator, ClassifierMixin):

    def __init__(self, classifier_constructor):
        self.classifier_constructor = classifier_constructor
        self.trained_classifier = None
        
    def fit(self, X, y):
        # convert X,y to a data
        assert len(X) == len(y)
        df = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
        assert np.sum(df.iloc[:,-1]-y) == 0
        train_data = monotonic_utils.raw_data_to_monotonic_input(df)
        import pdb
        return monotonic_predictor(self.classifier_constructor(train_data))
