#from sklearn.base import BaseEstimator, ClassifierMixin
import monotonic.monotonic.utils as monotonic_utils
import numpy as np
import pandas as pd
import extra_utils as caching

class monotonic_predictor(object):

    def __init__(self, horse):
        self.horse = horse

    def decision_function(self, X):
        test_data = monotonic_utils.data(hash(caching.get_hash(X)), range(len(X)), X, [None for i in xrange(len(X))])
        return np.array([self.horse(datum) for datum in test_data])

    def predict_proba(self, X):
        one_probs = self.decision_function(X)
        return np.array([1.0-one_probs,one_probs]).T

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
    
    def train_info(self):
        return self.horse.train_info()[['gamma','p','prob','support','overall_support','features_in_rule']].iloc[:-1,:]
        
        
class monotonic_fitter_from_constructor(object):

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

class monotonic_fitter(monotonic_fitter_from_constructor):

    def __init__(self, num_steps = 5000, min_supp = 5, max_clauses = 2, prior_length_mean = 8, prior_gamma_l_alpha = 1., prior_gamma_l_beta = 0.1, temperature = 1):
        import monotonic.monotonic.rule as rule
        import monotonic.monotonic.model as model
        import monotonic.monotonic.distributions as distributions
        import monotonic.monotonic.mcmc_step_fs as mcmc_step_fs
        import monotonic.monotonic.mcmc as mcmc
        import monotonic.monotonic.predictors as predictors
        rule_miner_f = rule.rule_miner_f(min_supp, max_clauses)
        rule_f_ls_given_L_dist_constructor = model.fixed_set_rule_f_ls_given_L_dist_constructor(rule_miner_f)
        L_dist = distributions.poisson_dist(prior_length_mean)
        gamma_ls_dist_alpha, gamma_ls_dist_beta = prior_gamma_l_alpha, prior_gamma_l_beta
        gamma_ls_given_L_dist = model.gamma_ls_given_L_dist(gamma_ls_dist_alpha, gamma_ls_dist_beta)
        accept_proposal_f = mcmc_step_fs.simulated_annealing_accept_proposal_f(mcmc_step_fs.constant_temperature_f(temperature))
        mcmc_step_f_constructors = [\
                                    mcmc_step_fs.generic_mcmc_step_f_constructor(mcmc_step_fs.rule_swap_only_mh_step_f, mcmc_step_fs.reduced_posterior_obj_f_constructor(), accept_proposal_f),\
                                    mcmc_step_fs.generic_mcmc_step_f_constructor(mcmc_step_fs.add_or_remove_rule_mh_step_f, mcmc_step_fs.reduced_posterior_obj_f_constructor(), accept_proposal_f),\
                                    mcmc_step_fs.generic_mcmc_step_f_constructor(mcmc_step_fs.replace_rule_mh_step_f, mcmc_step_fs.reduced_posterior_obj_f_constructor(), accept_proposal_f),\
                                    ]
        theta_dist_constructor = model.theta_dist_constructor(rule_f_ls_given_L_dist_constructor, gamma_ls_given_L_dist, L_dist)
        get_traces_f = mcmc.get_traces_f(theta_dist_constructor, mcmc_step_f_constructors, num_steps)
        my_predictor_constructor = predictors.simple_map_single_decision_list_predictor_constructor(get_traces_f)
        self.classifier_constructor = my_predictor_constructor
