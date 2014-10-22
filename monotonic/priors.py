import monotonic.monotonic.utils as monotonic_utils
import monotonic.monotonic.utils as utils
import scipy.stats
import monotonic.monotonic.distributions as distributions
import numpy as np

##### for p_gamma_ls_given_L_f #####

class simple_p_gamma_ls_given_L_f(distributions.infinite_vectorized_conditional_dist):

    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
        self.gamma_dist = distributions.gamma_dist(alpha, beta)
        self.truncated_gamma_dist = distributions.gamma_dist(alpha, beta, 1.0)
        
    def loglik(self, N, n, y, x):
        if n == N:
            return self.truncated_gamma_dist.loglik(x)
        else:
            return self.gamma_dist.loglik(x)
        
class get_simple_p_gamma_ls_given_L_f(monotonic_utils.f_base):

    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
    
    def __call__(self, data):
        return simple_p_gamma_ls_given_L_f(self.alpha, self.beta)
        
##### for p_L_f #####

get_poisson_p_L_f = distributions.poisson_dist

##### for p_rule_f #####

class simple_rule_f_dist(distributions.infinite_vectorized_dist):
    """
    multiplicative penalty.  doesn't matter for now
    """
    def __init__(self, c):
        self.c = c

    def loglik(self, N, n, rule_f):
        return self.c + np.log(rule_f.cardinality)

class get_simple_rule_f_dist(monotonic_utils.f_base):

    def __init__(self, c):
        self.c = c

    def __call__(self, data):
        return simple_rule_f(self.c)

##### for p_zeta_ns #####

class get_zeta_ns_dist(monotonic_utils.f_base):

    def __call__(self, data):
        return distributions.vectorized_dist(len(data), distributions.exp_dist(1.0))
