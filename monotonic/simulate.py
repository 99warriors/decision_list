import monotonic.monotonic.utils as monotonic_utils
import monotonic.monotonic.model as model
import numpy as np
import monotonic.monotonic.distributions as distributions
import scipy.stats
import monotonic.monotonic.rule as rule
import monotonic.monotonic.distributions as distributions
import monotonic.monotonic.model as model
    
def get_dummy_data(N, data_id):
    return monotonic_utils.data(data_id, range(N), np.array([None for i in xrange(N)]), np.array([None for i in xrange(N)]))

def simulate_theta_and_data_fixed_rule_f_ls_fixed_gamma_ls(N, L, match_prob, gamma_l):
    dummy_data = get_dummy_data(N)
    rule_f_ls = rule.get_hard_coded_rule_f(match_prob, L)(dummy_data)
    gamma_ls = np.array([gamma_l for i in xrange(L+1)])
    zeta_ns_dist = distributions.vectorized_iid_dist(len(dummy_data), distributions.exp_dist(1.0))
    fixed_theta_dist = model.theta_fixed_rule_f_ls_fixed_gamma_ls_dist(rule_f_ls, gamma_ls, zeta_ns_dist, dummy_data)
    simulated_theta_and_data = model.theta_and_data_joint_dist(fixed_theta_dist, model.data_given_theta_dist()).sample()
    return simulated_theta_and_data
