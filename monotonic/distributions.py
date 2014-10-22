import monotonic.monotonic.utils as monotonic_utils
import scipy.stats
import numpy as np

class dist(monotonic_utils.obj_base):

    def loglik(self, x):
        pass

    def sample(self):
        pass

class vectorized_dist(dist):

    def batch_loglik(self, xs):
        pass

    def batch_sample(self):
        pass

    def get(self, i):
        """
        only defined if component distributions are independent
        """
        pass
    
class conditional_dist(dist):

    def loglik(self, y, x):
        pass

    def sample(self, y):
        pass

class conditional_vectorized_dist(dist):

    def batch_loglik(self, given, xs):
        pass

    def batch_sample(self, given):
        pass

    def get(self, given, i):
        pass
            
class gamma_dist(dist):

    def __init__(self, alpha, beta, left_truncate = 0):
        self.alpha, self.beta = alpha, beta
        self.horse = scipy.stats.gamma(a=alpha, scale=1./beta)

    def sample(self):
        return self.horse.rvs()

    def loglik(self, x):
        return self.horse.logpdf(x)

class poisson_dist(dist):

    def __init__(self, rate, left_truncate = 0):
        self.rate = rate
        self.horse = scipy.stats.poisson(mu=rate)

    def sample(self):
        return self.horse.rvs()

    def loglik(self, x):
        return self.horse.logpmf(x)

class constant_dist(dist):

    def loglik(self, x):
        return 0
    
class exp_dist(dist):

    def __init__(self, rate):
        self.rate = rate
        self.horse = scipy.stats.expon(scale = 1./rate)

    def loglik(self, x):
        return self.horse.logpdf(x)

    def sample(self):
        return self.horse.rvs()
