import monotonic.monotonic.mcmc as mcmc
import monotonic.monotonic.utils as monotonic_utils
import numpy as np
import monotonic.monotonic.model as model
import itertools
import pandas as pd
import extra_utils as caching
import pdb

class classifier(monotonic_utils.f_base):

    def __call__(self, datum):
        raise NotImplementedError

class single_decision_list_predictor(classifier):

    def __init__(self, reduced_theta, train_data):
        assert len(reduced_theta.rule_f_ls) == len(reduced_theta.gamma_ls) - 1
        self.reduced_theta, self.train_data = reduced_theta, train_data

    def __call__(self, datum):
        return self.reduced_theta.p_ls[self.reduced_theta.get_z_n(datum)]

    def train_info(self):
        return model.theta_and_data(model.theta(self.reduced_theta.rule_f_ls, self.reduced_theta.gamma_ls, None, None, self.train_data), self.train_data).informative_df()


        
class simple_map_single_decision_list_predictor_constructor(monotonic_utils.f_base):

    def __init__(self, get_traces_f):
        self.get_traces_f = get_traces_f

#    @caching.read_method_decorator(caching.read_pickle, custom_get_path, 'pickle')
#    @caching.write_method_decorator(caching.write_pickle, custom_get_path, 'pickle')
#    @caching.default_write_method_decorator
    def __call__(self, data):

        traces = self.get_traces_f(data)
        thetas = traces.get_thetas(lambda diff_f: True)
        theta_and_datas = [model.theta_and_data(theta, data) for theta in thetas]
        
        theta_dist = self.get_traces_f.theta_dist_constructor(data)
        theta_and_data_dist = model.theta_and_data_dist(theta_dist, model.data_given_theta_dist())
        
        posteriors = [theta_and_data_dist.reduced_loglik(theta_and_data) for theta_and_data in theta_and_datas]
        best_theta, best_loglik = max(itertools.izip(thetas, posteriors), key = lambda (theta,posterior): posterior)
        return single_decision_list_predictor(model.reduced_theta(best_theta.rule_f_ls, best_theta.gamma_ls), data)

def map_list_then_map_params_helper(get_traces_f, thetas, data):

    rule_f_idx_ls = pd.Series([theta.rule_f_idx_ls for theta in thetas])
    map_rule_f_idx = rule_f_idx_ls.value_counts().index[0]
    filtered_thetas = [theta for theta in thetas if theta.rule_f_idx_ls == map_rule_f_idx]
    filtered_theta_and_datas = [model.theta_and_data(theta, data) for theta in filtered_thetas]

    theta_dist = get_traces_f.theta_dist_constructor(data)
    theta_and_data_dist = model.theta_and_data_dist(theta_dist, model.data_given_theta_dist())

    filtered_posteriors = [theta_and_data_dist.reduced_loglik(theta_and_data) for theta_and_data in filtered_theta_and_datas]

    best_theta, best_loglik = max(itertools.izip(filtered_thetas, filtered_posteriors), key = lambda (theta,posterior): posterior)

    return best_theta, best_loglik
        
class map_list_then_map_params_predictor_constructor(monotonic_utils.f_base):

    def __init__(self, get_traces_f):
        self.get_traces_f = get_traces_f
    
    def __call__(self, data):

        traces = self.get_traces_f(data)
        thetas = traces.get_thetas(lambda diff_f: True)

        best_theta, best_loglik = map_list_then_map_params_helper(self.get_traces_f, thetas, data)
        
        return single_decision_list_predictor(model.reduced_theta(best_theta.rule_f_ls, best_theta.gamma_ls), data)
