import monotonic.monotonic.mcmc as mcmc
from collections import namedtuple
#import random
import monotonic.monotonic.model as model
import copy
import scipy.stats
import numpy as np
import monotonic.monotonic.utils as monotonic_utils
import pdb
import itertools
import copy

debug_prob = -1
strong_debug = -1

##### mcmc_step_f constructors and requisite constructors #####

class metropolis_hastings_mcmc_step_f(mcmc.mcmc_step_f):

    def __call__(self, theta_and_data):
        raise NotImplementedError

class generic_mcmc_step_f_constructor(monotonic_utils.f_base):

    def __init__(self, cls, obj_f_constructor, accept_proposal_f):
        self.cls, self.obj_f_constructor, self.accept_proposal_f = cls, obj_f_constructor, accept_proposal_f
    
    def __call__(self, theta_and_data_dist):
        return self.cls(theta_and_data_dist, self.obj_f_constructor(theta_and_data_dist), self.accept_proposal_f)

class reduced_posterior_obj_f_constructor(monotonic_utils.f_base):

    def __call__(self, theta_and_data_dist):
        return lambda theta_and_data: theta_and_data_dist.reduced_theta_loglik(theta_and_data) + theta_and_data_dist.reduced_data_given_theta_loglik(theta_and_data)
        
class generic_gibbs_mcmc_step_f_constructor(monotonic_utils.f_base):

    def __init__(self, cls):
        self.cls = cls

    def __call__(self, theta_and_data_dist):
        return self.cls(theta_and_data_dist)

class simulated_annealing_accept_proposal_f(monotonic_utils.f_base):

    def __init__(self, temperature_f):
        self.temperature_f = temperature_f
        self.step_count = 0

    def __call__(self, old_loglik, new_loglik):
        assert np.isfinite(old_loglik)
        assert np.isfinite(new_loglik)
        temperature = self.temperature_f(self.step_count)
        self.step_count += 1
        accept_prob = np.exp(-(old_loglik - new_loglik) / temperature)
        return np.random.random() < accept_prob

class constant_temperature_f(monotonic_utils.f_base):

    def __init__(self, the_temperature):
        self.the_temperature = the_temperature

    def __call__(self, step_count):
        return self.the_temperature
    
##### gibbs steps #####

class gibbs_diff_f(mcmc.diff_f):
    """
    assumes variable can be refered to using __dict__
    TODO: what cannot refer to variable using __dict__?
    """
    def __init__(self, param_idx, new_val, _accepted):
        self.param_idx, self.new_val, self._accepted = param_idx, new_val, _accepted

    @property
    def accepted(self):
        return True

    def make_change(self, theta):
        try:
            theta[self.param_idx] = self.new_val
        except (KeyError, TypeError):
            for param_idx, new_val in itertools.izip(self.param_idx, self.new_val):
                theta[param_idx] = new_val
                
class gamma_ls_gibbs_step_f(mcmc.mcmc_step_f):

    name = 'gamma_ls_gibbs_step_f'
    
    def __init__(self, theta_and_data_dist):
        self.theta_and_data_dist = theta_and_data_dist
    
    def __call__(self, theta_and_data):

        assert theta_and_data.theta.L+1 == len(theta_and_data.theta.gamma_ls)
        
        def reverse_cumsum(v):
            return [y for y in reversed(np.cumsum([x for x in reversed(v)]))]

        old_gamma_ls = copy.deepcopy(theta_and_data.theta.gamma_ls)

        for l in range(0,theta_and_data.theta.L+1):
            c_l = np.exp(reverse_cumsum(np.log(theta_and_data.theta.gamma_ls)) - np.log(theta_and_data.theta.gamma_ls[l]))
            c_l[(l+1):] = 0
            gamma_l_dist = self.theta_and_data_dist.theta_dist.gamma_ls_given_L_dist.get(theta_and_data.theta.L+1, l)
            gamma_l_dist_alpha, gamma_l_dist_beta = gamma_l_dist.alpha, gamma_l_dist.beta
            gamma_l_gibbs_dist_alpha = gamma_l_dist_alpha + (theta_and_data.theta.z_ns <= l).dot(theta_and_data.theta.w_ns)
            gamma_l_gibbs_dist_beta = gamma_l_dist_beta + c_l[theta_and_data.theta.z_ns].dot(theta_and_data.theta.zeta_ns)

            if l == theta_and_data.theta.L:
                theta_and_data.theta.gamma_ls[l] = monotonic_utils.sample_truncated_gamma(gamma_l_gibbs_dist_alpha, gamma_l_gibbs_dist_beta, 0.001)
            else:
                theta_and_data.theta.gamma_ls[l] = monotonic_utils.sample_truncated_gamma(gamma_l_gibbs_dist_alpha, gamma_l_gibbs_dist_beta, 1.)
                
        new_gamma_ls = copy.deepcopy(theta_and_data.theta.gamma_ls)
        theta_and_data.theta.gamma_ls = old_gamma_ls
        return gibbs_diff_f('gamma_ls', new_gamma_ls, np.ones(theta_and_data.theta.L+1, dtype=bool))

class w_ns_gibbs_step_f(mcmc.mcmc_step_f):

    name = 'w_ns_gibbs_step_f'
    
    def __init__(self, theta_and_data_dist):
        self.theta_and_data_dist = theta_and_data_dist
        
    def __call__(self, theta_and_data):
        rates = theta_and_data.theta.zeta_ns * theta_and_data.theta.v_ns
        new_w_ns = monotonic_utils.vectorized_zero_truncated_poisson_sample(rates)
        new_w_ns[theta_and_data.data.y_ns==0] = 0
        return gibbs_diff_f('w_ns', new_w_ns, np.ones(theta_and_data.theta.N, dtype=bool))

class zeta_ns_gibbs_step_f(mcmc.mcmc_step_f):

    name = 'zeta_ns_gibbs_step_f'
    
    def __init__(self, theta_and_data_dist):
        self.theta_and_data_dist = theta_and_data_dist

    def __call__(self, theta_and_data):
        return gibbs_diff_f('zeta_ns', scipy.stats.gamma(a=theta_and_data.theta.w_ns + 1, scale=1.0 / (theta_and_data.theta.v_ns + 1)).rvs(), np.ones(theta_and_data.theta.N, dtype=bool))

class w_ns_zeta_ns_gibbs_step_f(mcmc.mcmc_step_f):

    name = 'w_ns_zeta_ns_gibbs_step_f'

    def __init__(self, theta_and_data_dist):
        self.theta_and_data_dist = theta_and_data_dist

    def __call__(self, theta_and_data):
        w_ns = scipy.stats.geom(p=1.0 / (1 + theta_and_data.theta.v_ns)).rvs()
        w_ns[theta_and_data.data.y_ns==0] = 0
        alpha_ns = 1.0 + w_ns
        beta_ns = 1.0 + theta_and_data.theta.v_ns
        zeta_ns = scipy.stats.gamma(a=alpha_ns, scale=1./beta_ns).rvs()
        return gibbs_diff_f(['w_ns','zeta_ns'],[w_ns,zeta_ns],np.ones(theta_and_data.theta.N, dtype=bool))
    
# changing the rule_f list

class replace_rule_mh_diff_f(mcmc.diff_f):

    def __init__(self, replace_pos, replace_rule_f, new_gamma_ls, _accepted):
        self.replace_pos, self.replace_rule_f, self.new_gamma_ls, self._accepted = replace_pos, replace_rule_f, new_gamma_ls, _accepted

    @property
    def param_idx(self):
        return 'rule_f_ls'
        
    @property
    def accepted(self):
        return self._accepted

    def make_change(self, theta):
        theta.rule_f_ls[self.replace_pos] = self.replace_rule_f
        theta.gamma_ls = self.new_gamma_ls
        
class replace_rule_mh_step_f(mcmc.mcmc_step_f):
            
    name = 'replace_rule_mh_step_f'

    def __init__(self, theta_and_data_dist, obj_f, accept_proposal_f):
        self.obj_f, self.accept_proposal_f = obj_f, accept_proposal_f
        self.theta_and_data_dist = theta_and_data_dist
        
    def __call__(self, theta_and_data):

        old_rule_f_idx_ls = theta_and_data.theta.rule_f_idx_ls
        # choose position and rule to replace

        # use reduced model or not
        reduced = True
        # right now, only have 1 print option.  this may change
        debug = np.random.random() < debug_prob

        # the probs returned depend on whether using reduced model or not
        def get_probs(_theta_and_data):
            if reduced:
                return self.theta_and_data_dist.reduced_theta_loglik(theta_and_data), self.theta_and_data_dist.reduced_data_given_theta_loglik(theta_and_data)
            else:
                return self.theta_and_data_dist.theta_dist.loglik(theta_and_data.theta), self.theta_and_data_dist.data_given_theta_dist.loglik(theta_and_data.theta, theta_and_data.data)
                
        # modify this as decisions are being made
        log_q_ratio = 0.

        # get old probs
        # old_log_theta_prob, old_log_theta_given_theta_prob = get_probs(theta_and_data)

        old_loglik = self.obj_f(theta_and_data)
        
        def q_replace_pos():
            replace_pos = np.random.randint(0, theta_and_data.theta.L)
            return replace_pos, 0

        replace_pos, replace_pos_log_q_ratio = q_replace_pos()
        log_q_ratio += replace_pos_log_q_ratio
        
        def q_replace_rule_f():
#            print theta_and_data.theta.rule_f_ls
            replace_rule_f = self.theta_and_data_dist.theta_dist.rule_f_ls_given_L_dist.iterative_sample(theta_and_data.theta.rule_f_ls)
            # TODO: the iterative sampling prob forwards and backwards 
            return replace_rule_f, 0

        replacement_rule_f, replace_rule_f_log_q_ratio = q_replace_rule_f()
        log_q_ratio += replace_rule_f_log_q_ratio

#        print replacement_rule_f
        
        def q_new_gamma_ls_optimize_all():
            replaced_rule_f = theta_and_data.theta.rule_f_ls[replace_pos]
            theta_and_data.theta.rule_f_ls[replace_pos] = replacement_rule_f
            new_gamma_ls = theta_and_data.greedy_optimal_gamma_ls()
            theta_and_data.theta.rule_f_ls[replace_pos] = replaced_rule_f
            return new_gamma_ls, 0

        q_new_gamma_ls = q_new_gamma_ls_optimize_all

        new_gamma_ls, new_gamma_ls_log_q_ratio = q_new_gamma_ls()
        log_q_ratio += new_gamma_ls_log_q_ratio

        
        if debug:
            print '\nreplace rule at', replace_pos, 'out of', theta_and_data.theta.L
            print 'BEFORE'
            print theta_and_data.informative_df()
                
        # all the decisions have been made.  now make the actual changes
        replaced_rule_f = theta_and_data.theta.rule_f_ls[replace_pos]
        theta_and_data.theta.rule_f_ls[replace_pos] = replacement_rule_f
        old_gamma_ls = theta_and_data.theta.gamma_ls
        theta_and_data.theta.gamma_ls = new_gamma_ls
        
        # decide whether to change
#        new_log_theta_prob, new_log_theta_given_theta_prob = get_probs(theta_and_data)
        new_loglik = self.obj_f(theta_and_data)
        accept = self.accept_proposal_f(old_loglik, new_loglik)
        
 #       log_accept_prob = log_q_ratio + (new_log_theta_prob + new_log_theta_given_theta_prob) - (old_log_theta_prob + old_log_theta_given_theta_prob)
 #       accept = np.random.random() < np.exp(log_accept_prob)

        if old_rule_f_idx_ls == (0,1,2,3,4) and accept:
            theta_and_data.theta.rule_f_ls[replace_pos] = replaced_rule_f
            theta_and_data.theta.gamma_ls = old_gamma_ls
            print 'BEFORE LEAVING THE BEST'
            print theta_and_data.informative_df()
            theta_and_data.theta.rule_f_ls[replace_pos] = replacement_rule_f
            theta_and_data.theta.gamma_ls = new_gamma_ls
            debug = True
        
        if debug:
            print 'AFTER'
            print theta_and_data.informative_df()
            print 'old_loglik: %.2f new_loglik %.2f' % (old_loglik, new_loglik)
#            print 'log_q_ratio', log_q_ratio
#            print 'new_log_theta_prob', new_log_theta_prob
#            print 'old_log_theta_prob', old_log_theta_prob
#            print 'log_theta_prob_diff', new_log_theta_prob - old_log_theta_prob
#            print 'new_likelihood', new_log_theta_given_theta_prob
#            print 'old_likelihood', old_log_theta_given_theta_prob
#            print 'log_likelihood_diff', new_log_theta_given_theta_prob - old_log_theta_given_theta_prob
#            print 'accept_prob', np.exp(log_accept_prob)
        
        # undo the change, because this should not actual modify things
        theta_and_data.theta.rule_f_ls[replace_pos] = replaced_rule_f
        theta_and_data.theta.gamma_ls = old_gamma_ls
        
        return replace_rule_mh_diff_f(replace_pos, replacement_rule_f, new_gamma_ls, accept)
    
class rule_swap_only_mh_diff_f(mcmc.diff_f):

    def __init__(self, (rule_f_a_idx, rule_f_a), (rule_f_b_idx, rule_f_b), new_gamma_ls, _accepted):
        (self.rule_f_a_idx, self.rule_f_a), (self.rule_f_b_idx, self.rule_f_b), self.new_gamma_ls, self._accepted = (rule_f_a_idx, rule_f_a), (rule_f_b_idx, rule_f_b), new_gamma_ls, _accepted

    @property
    def param_idx(self):
        return 'swap rule_f_ls'
         
    @property
    def accepted(self):
        return self._accepted
        
    def make_change(self, theta):
#        assert theta.rule_f_ls[self.rule_f_a_idx] == self.rule_f_a
#        assert theta.rule_f_ls[self.rule_f_b_idx] == self.rule_f_b
        import copy
        theta.rule_f_ls = copy.copy(theta.rule_f_ls)
        theta.gamma_ls = self.new_gamma_ls
        monotonic_utils.swap_list_items(theta.rule_f_ls, self.rule_f_a_idx, self.rule_f_b_idx)
    
class rule_swap_only_mh_step_f(mcmc.mcmc_step_f):
    
    name = 'rule_swap_only_mh_step_f'

    def __init__(self, theta_and_data_dist, obj_f, accept_proposal_f):
        self.obj_f, self.accept_proposal_f = obj_f, accept_proposal_f
        self.theta_and_data_dist = theta_and_data_dist

    def __call__(self, theta_and_data):

        old_rule_f_idx_ls = theta_and_data.theta.rule_f_idx_ls
            
        if theta_and_data.theta.L >= 2:
        
            # use reduced model or not
            reduced = True
            # right now, only have 1 print option.  this may change
            debug = np.random.random() < debug_prob

            # the probs returned depend on whether using reduced model or not
            def get_probs(_theta_and_data):
                if reduced:
                    return self.theta_and_data_dist.reduced_theta_loglik(theta_and_data), self.theta_and_data_dist.reduced_data_given_theta_loglik(theta_and_data)
                else:
                    return self.theta_and_data_dist.theta_dist.loglik(theta_and_data.theta), self.theta_and_data_dist.data_given_theta_dist.loglik(theta_and_data.theta, theta_and_data.data)

            # modify this as decisions are being made
            log_q_ratio = 0.

            # get old probs
            #old_log_theta_prob, old_log_theta_given_theta_prob = get_probs(theta_and_data)

            old_loglik = self.obj_f(theta_and_data)
            
            # decide which rules to swap
            def q_swap_idx():
                if theta_and_data.theta.L == 1:
                    idx_a, idx_b = 0, 0
                else:
                    try:
                        idx_a, idx_b = np.random.choice(range(theta_and_data.theta.L), 2, replace=False)
                    except:
                        import pdb
                        pdb.set_trace()
                        print theta_and_data.theta.L
                return (idx_a, idx_b), 0

            
            
            (idx_a, idx_b), swap_idx_log_q_ratio = q_swap_idx()
            log_q_ratio += swap_idx_log_q_ratio


            def q_new_gamma_ls_optimize_all():
                monotonic_utils.swap_list_items(theta_and_data.theta.rule_f_ls, idx_a, idx_b)
                new_gamma_ls = theta_and_data.greedy_optimal_gamma_ls()
                monotonic_utils.swap_list_items(theta_and_data.theta.rule_f_ls, idx_a, idx_b)
                return new_gamma_ls, 0

            q_new_gamma_ls = q_new_gamma_ls_optimize_all
            
            new_gamma_ls, new_gamma_ls_log_q_ratio = q_new_gamma_ls()
            log_q_ratio += new_gamma_ls_log_q_ratio

            
            if debug:
                print '\nswapping', idx_a, idx_b, 'out of', theta_and_data.theta.L
                print 'BEFORE'
                print theta_and_data.informative_df()

            # all the decisions have been made.  now make the actual changes              
            monotonic_utils.swap_list_items(theta_and_data.theta.rule_f_ls, idx_a, idx_b)
            old_gamma_ls = theta_and_data.theta.gamma_ls
            theta_and_data.theta.gamma_ls = new_gamma_ls

            
            # decide whether to change

            new_loglik = self.obj_f(theta_and_data)
            accept = self.accept_proposal_f(old_loglik, new_loglik)
            
#            new_log_theta_prob, new_log_theta_given_theta_prob = get_probs(theta_and_data)
#            log_accept_prob = log_q_ratio + (new_log_theta_prob + new_log_theta_given_theta_prob) - (old_log_theta_prob + old_log_theta_given_theta_prob)
#            accept = np.random.random() < np.exp(log_accept_prob)

            if old_rule_f_idx_ls == (0,1,2,3,4) and accept:
                monotonic_utils.swap_list_items(theta_and_data.theta.rule_f_ls, idx_a, idx_b)
                theta_and_data.theta.gamma_ls = old_gamma_ls
                print 'BEFORE LEAVING THE BEST'
                print theta_and_data.informative_df()
                monotonic_utils.swap_list_items(theta_and_data.theta.rule_f_ls, idx_a, idx_b)
                theta_and_data.theta.gamma_ls = new_gamma_ls
                debug = True
            
            if debug:
                print 'AFTER'
                print theta_and_data.informative_df()
                print 'old_loglik: %.2f new_loglik %.2f' % (old_loglik, new_loglik)
#                print 'log_q_ratio', log_q_ratio
#                print 'new_log_theta_prob', new_log_theta_prob
#                print 'old_log_theta_prob', old_log_theta_prob
#                print 'log_theta_prob_diff', new_log_theta_prob - old_log_theta_prob
#                print 'new_likelihood', new_log_theta_given_theta_prob
#                print 'old_likelihood', old_log_theta_given_theta_prob
#                print 'log_likelihood_diff', new_log_theta_given_theta_prob - old_log_theta_given_theta_prob
#                print 'accept_prob', np.exp(log_accept_prob)

            # undo the change, because this should not actual modify things
            monotonic_utils.swap_list_items(theta_and_data.theta.rule_f_ls, idx_a, idx_b)
            theta_and_data.theta.gamma_ls = old_gamma_ls
            return rule_swap_only_mh_diff_f((idx_a, theta_and_data.theta.rule_f_ls[idx_a]), (idx_b, theta_and_data.theta.rule_f_ls[idx_b]), new_gamma_ls, accept)
        else:
            return rule_swap_only_mh_diff_f((None, None), (None, None), None, False)

"""
below is old stuff

    
        
            if reduced:
                new_log_theta_prob = self.theta_and_data_dist.theta_dist.reduced_loglik(theta_and_data.theta)
            else:
                new_log_theta_prob = self.theta_and_data_dist.theta_dist.loglik(theta_and_data.theta)

                
            log_theta_prob_ratio = new_log_theta_prob - old_log_theta_prob
            #new_log_data_given_theta_prob = self.theta_and_data_dist.data_given_theta_dist.loglik(theta_and_data.theta, theta_and_data.data)


            new_log_data_given_theta_prob = self.theta_and_data_dist.data_given_theta_dist.original_likelihood_dist.loglik(theta_and_data)


            
            log_data_given_theta_prob_ratio = new_log_data_given_theta_prob - old_log_data_given_theta_prob
            
            log_acceptance_prob = log_Q_prob_ratio + log_theta_prob_ratio + log_data_given_theta_prob_ratio

            if debug: print '\nswap', log_acceptance_prob, np.exp(log_acceptance_prob), '\n'
            
            accept = np.random.random() < np.exp(log_acceptance_prob)
    
            monotonic_utils.swap_list_items(theta_and_data.theta.rule_f_ls, idx_a, idx_b)
            return rule_swap_only_mh_diff_f((idx_a, theta_and_data.theta.rule_f_ls[idx_a]), (idx_b, theta_and_data.theta.rule_f_ls[idx_b]), accept)
        else:
            return rule_swap_only_mh_diff_f((None,None), (None,None), False)

"""
                        
class add_or_remove_rule_mh_diff_f(mcmc.diff_f):

    def __init__(self, change_type, _accepted, pos, (added_rule_f, new_gamma_ls) = (None, None)):
        self.change_type, self._accepted, self.pos, (self.added_rule_f, self.new_gamma_ls) = change_type, _accepted, pos, (added_rule_f, new_gamma_ls)

    @property
    def param_idx(self):
        return 'add rule_f_ls' if self.change_type == add_or_remove_rule_mh_step_f.ADD else 'remove add_f_ls'
         
    @property
    def accepted(self):
        return self._accepted
        
    def make_change(self, theta):
        if self.change_type == add_or_remove_rule_mh_step_f.ADD:
            theta.rule_f_ls.insert(self.pos, self.added_rule_f)
            theta.gamma_ls = self.new_gamma_ls
#            theta.gamma_ls = np.insert(theta.gamma_ls, self.pos, self.added_gamma)
        elif self.change_type == add_or_remove_rule_mh_step_f.REMOVE:
            theta.rule_f_ls.pop(self.pos)
            theta.gamma_ls = self.new_gamma_ls
#            theta.gamma_ls = np.delete(theta.gamma_ls, self.pos)
            
    
class add_or_remove_rule_mh_step_f(mcmc.mcmc_step_f):
    """
    can either:
    insert a rule before l-th rule (l = 0...L-1)
    remove l-th rule (l = 0...L-1)
    move l-th rule (l = 0...L-1) to before l'-th rule (l = 0...L)
    """
    ADD = 0
    REMOVE = 1

    name = 'add_or_remove_rule_mh_step_f'

    def __init__(self, theta_and_data_dist, obj_f, accept_proposal_f):
        self.obj_f, self.accept_proposal_f = obj_f, accept_proposal_f
        self.theta_and_data_dist = theta_and_data_dist
    
    def __call__(self, theta_and_data):

        old_rule_f_idx_ls = theta_and_data.theta.rule_f_idx_ls
        
        # use reduced model or not
        reduced = True
        # inserts/removals at end of list (aka position L) cause problems.  whether to allow inserts there
        allow_end_changes = True
        # right now, only have 1 print option.  this may change
        debug = np.random.random() < debug_prob
        # whether q_gamma to use
        

        # the probs returned depend on whether using reduced model or not
        def get_probs(_theta_and_data):
            if reduced:
                return self.theta_and_data_dist.reduced_theta_loglik(theta_and_data), self.theta_and_data_dist.reduced_data_given_theta_loglik(theta_and_data)
            else:
                return self.theta_and_data_dist.theta_dist.loglik(theta_and_data.theta), self.theta_and_data_dist.data_given_theta_dist.loglik(theta_and_data.theta, theta_and_data.data)
                
        # modify this as decisions are being made
        log_q_ratio = 0.

        # get old probs
        old_loglik = self.obj_f(theta_and_data)
#        old_log_theta_prob, old_log_theta_given_theta_prob = get_probs(theta_and_data)
        
        # decide whether to add or remove node
        def q_add_or_remove():

            if theta_and_data.theta.L == 1:
                return add_or_remove_rule_mh_step_f.ADD, 0.
            elif theta_and_data.theta.L == len(self.theta_and_data_dist.theta_dist.rule_f_ls_given_L_dist.possible_rule_fs):
                return add_or_remove_rule_mh_step_f.REMOVE, 0.
            else:
                add_prob = 0.5
                if scipy.stats.uniform.rvs() < add_prob:
                    _log_q_ratio = np.log(add_prob) - np.log(1.-add_prob)
                    return add_or_remove_rule_mh_step_f.ADD, _log_q_ratio
                else:
                    _log_q_ratio = np.log(1.-add_prob) - np.log(add_prob)
                    return add_or_remove_rule_mh_step_f.REMOVE, _log_q_ratio

        add_or_remove, add_or_remove_log_q_ratio = q_add_or_remove()
        log_q_ratio += add_or_remove_log_q_ratio
        
                
        if add_or_remove == add_or_remove_rule_mh_step_f.ADD:

            # decide the insert position
            def q_insert_pos():
                if allow_end_changes:
                    insert_pos = np.random.randint(0, theta_and_data.theta.L+1)
                    return insert_pos, 0
                else:
                    insert_pos = np.random.randint(0, theta_and_data.theta.L)
                    return insert_pos, 0

            insert_pos, insert_pos_log_q_ratio = q_insert_pos()
            log_q_ratio += insert_pos_log_q_ratio
                
            # decide the rule to insert
            def q_insert_rule():
                # pretend the rule was already there, just was not being used.  thus not an actual change
                insert_rule = self.theta_and_data_dist.theta_dist.rule_f_ls_given_L_dist.iterative_sample(theta_and_data.theta.rule_f_ls)
                return insert_rule, 0

            insert_rule_f, insert_rule_f_log_q_ratio = q_insert_rule()
            log_q_ratio += insert_rule_f_log_q_ratio
            
            # decide the new gamma_ls, using simplest method: drawn a single gamma from prior, and set gamma_ls[insert_pos] to it
            def q_new_gamma_ls_simple():
                # pretend the gamma value was already there, just was not being used.  thus not an actual change
                new_gamma_ls = copy.copy(theta_and_data.theta.gamma_ls)
                if allow_end_changes:
                    insert_gamma = self.theta_and_data_dist.theta_dist.gamma_ls_given_L_dist.iterative_sample(theta_and_data.theta.gamma_ls)
                    #raise NotImplementedError
                else:
                    insert_gamma = self.theta_and_data_dist.theta_dist.gamma_ls_given_L_dist.iterative_sample(theta_and_data.theta.gamma_ls)
                new_gamma_ls = np.insert(new_gamma_ls, insert_pos, insert_gamma)
                return new_gamma_ls, 0

            def q_new_gamma_ls_optimize_all():
                theta_and_data.theta.rule_f_ls.insert(insert_pos, insert_rule_f)
                ans = theta_and_data.greedy_optimal_gamma_ls(), 0
                theta_and_data.theta.rule_f_ls.pop(insert_pos)
                return ans

            def q_new_gamma_ls_hybrid():
                if np.random.random() < 0.5:
#                if reduce(lambda (prev,ok), x: (x,x>=prev and ok), theta_and_data.data_p_ls, (0,True)):
                    return q_new_gamma_ls_optimize_all()
                else:
                    return q_new_gamma_ls_simple()
                
                            
            def q_new_gamma_ls_pair():
                # this defines reversible jump step.  code outside of this fxn doesn't know about reversible jump
                # ensures that product of the new gamma and one below it equals the original gamma

                if insert_pos == 0:
                    new_gamma = self.theta_and_data_dist.theta_dist.gamma_ls_given_L_dist.iterative_sample(theta_and_data.theta.gamma_ls)
                    new_gamma_ls = copy.copy(theta_and_data.theta.gamma_ls)
                    new_gamma_ls = np.insert(new_gamma_ls, 0, new_gamma)
                    return new_gamma_ls, 0
                
                target_gamma = theta_and_data.theta.gamma_ls[insert_pos-1]
                
                # insert the new rule just to see what the probability at the new node would be
                theta_and_data.theta.rule_f_ls.insert(insert_pos, insert_rule_f)
                below_empirical_p = theta_and_data.data_p_ls[insert_pos]
                theta_and_data.theta.rule_f_ls.pop(insert_pos)
#                below_gamma = below_empirical_p / theta_and_data.theta.v_ls[insert_pos]
                try:
                    below_gamma = np.exp(monotonic_utils.logit(below_empirical_p)) / theta_and_data.theta.v_ls[insert_pos]
                except IndexError:
                    print 'bottom', insert_pos
                    below_gamma = np.exp(monotonic_utils.logit(below_empirical_p))

                    
                assert target_gamma > 1.

                if below_gamma < 1.:
                    below_gamma = 1.
                
                #below_gamma = np.random.uniform(1., target_gamma)# if insert_pos != len(theta_and_data.theta.gamma_ls) - 1 else np.random.uniform(0., target_gamma)
                above_gamma = target_gamma / below_gamma

                if False and (above_gamma < 1. or below_gamma < 1 or target_gamma < 1):
                    print 'target', target_gamma, 'above', above_gamma, 'below', below_gamma
                    print 'insert pos:', insert_pos
                    print 'before'
                    print theta_and_data.informative_df()
                    theta_and_data.theta.rule_f_ls.insert(insert_pos, insert_rule_f)
                    theta_and_data.theta.gamma_ls = np.insert(theta_and_data.theta.gamma_ls, insert_pos, below_gamma)
                    theta_and_data.theta.gamma_ls[insert_pos-1] = above_gamma
                    print 'after'
                    print theta_and_data.informative_df()
                    pdb.set_trace()
                
                new_gamma_ls = copy.deepcopy(theta_and_data.theta.gamma_ls)
                new_gamma_ls = np.insert(new_gamma_ls, insert_pos-1, above_gamma)

                """
                if below_gamma < .999:
                    print below_gamma, 'below'
                    pdb.set_trace()
                
                if above_gamma < .999:
                    print above_gamma, 'above'
                    pdb.set_trace()
                """
                assert insert_pos < len(new_gamma_ls)-1
                new_gamma_ls[insert_pos] = below_gamma
                return new_gamma_ls, 0
                
                u = np.random.random()
                log_target_gamma = np.log(theta_and_data.theta.gamma_ls[insert_pos])
                log_C = np.log(u) - np.log(1-u)
#                print 'old length', len(theta_and_data.theta.gamma_ls)
                new_gamma_ls = copy.copy(theta_and_data.theta.gamma_ls)
                new_gamma_ls = np.insert(new_gamma_ls, insert_pos, np.exp((log_target_gamma - log_C) / 2.))
                new_gamma_ls[insert_pos+1] = new_gamma_ls[insert_pos] * np.exp(log_C)
                assert np.abs(log_target_gamma - (np.log(new_gamma_ls[insert_pos]) + np.log(new_gamma_ls[insert_pos+1]))) < .0001

                return new_gamma_ls, 0

            # specify which gamma_ls proposal to use, and decide on the proposal
#            q_new_gamma_ls = q_new_gamma_ls_hybrid
            q_new_gamma_ls = q_new_gamma_ls_optimize_all
#            q_new_gamma_ls = q_new_gamma_ls_pair
#            q_new_gamma_ls = q_new_gamma_ls_simple
            new_gamma_ls, new_gamma_ls_log_q_ratio = q_new_gamma_ls()
            log_q_ratio += new_gamma_ls_log_q_ratio

            if tuple([x[1] for x in theta_and_data.theta.rule_f_idx_ls]) == (0,1,2,3) and insert_pos == 4 and insert_rule_f.idx[1] == 4 and np.random.random() < strong_debug:

                debug = True

            if theta_and_data.theta.rule_f_idx_ls == (0,1,2,3,4) and np.random.random() < strong_debug:
                debug = True
            """
            if insert_rule_f.idx == 4:
                debug = True

            if np.random.random() < -1:
                debug = True
            """
                
            if debug:
                print '\nadd rule at', insert_pos, 'out of', theta_and_data.theta.L, 'with new gamma', new_gamma_ls
                print 'BEFORE'
                print theta_and_data.informative_df()
                
            # all the decisions have been made.  now make the actual changes
            theta_and_data.theta.rule_f_ls.insert(insert_pos, insert_rule_f)
            old_gamma_ls = theta_and_data.theta.gamma_ls
            theta_and_data.theta.gamma_ls = new_gamma_ls
            
            # decide whether to change
            new_loglik = self.obj_f(theta_and_data)
            accept = self.accept_proposal_f(old_loglik, new_loglik)
#            new_log_theta_prob, new_log_theta_given_theta_prob = get_probs(theta_and_data)
#            log_accept_prob = log_q_ratio + (new_log_theta_prob + new_log_theta_given_theta_prob) - (old_log_theta_prob + old_log_theta_given_theta_prob)
#            accept = np.random.random() < np.exp(log_accept_prob)

            if old_rule_f_idx_ls == (0,1,2,3,4) and accept:
                theta_and_data.theta.rule_f_ls.pop(insert_pos)
                theta_and_data.theta.gamma_ls = old_gamma_ls
                print 'BEFORE LEAVING THE BEST'
                print theta_and_data.informative_df()
                theta_and_data.theta.rule_f_ls.insert(insert_pos, insert_rule_f)
                theta_and_data.theta.gamma_ls = new_gamma_ls
                debug = True
            
            if debug:
                print 'AFTER'
                print theta_and_data.informative_df()
                print 'old_loglik: %.2f new_loglik %.2f' % (old_loglik, new_loglik)
                print 'WITH OPTIMIZED GAMMAS'
                older_gamma_ls = copy.copy(theta_and_data.theta.gamma_ls)
                theta_and_data.theta.gamma_ls = theta_and_data.greedy_optimal_gamma_ls()
                print theta_and_data.theta.rule_f_ls
                print theta_and_data.informative_df()
                print 'okokoko', self.theta_and_data_dist.reduced_data_given_theta_loglik(theta_and_data)
                print 'old', old_loglik, 'new', new_loglik
                pdb.set_trace()
                theta_and_data.theta.gamma_ls = older_gamma_ls
#                print 'log_q_ratio', log_q_ratio
#                print 'new_log_theta_prob', new_log_theta_prob
#                print 'old_log_theta_prob', old_log_theta_prob
#                print 'log_theta_prob_diff', new_log_theta_prob - old_log_theta_prob
#                print 'new_likelihood', new_log_theta_given_theta_prob
#                print 'old_likelihood', old_log_theta_given_theta_prob
#                print 'log_likelihood_diff', new_log_theta_given_theta_prob - old_log_theta_given_theta_prob
#                print 'accept_prob', np.exp(log_accept_prob)
                
                
            # undo the change, because this should not actual modify things
            theta_and_data.theta.rule_f_ls.pop(insert_pos)
            theta_and_data.theta.gamma_ls = old_gamma_ls

            return add_or_remove_rule_mh_diff_f(add_or_remove_rule_mh_step_f.ADD, accept, insert_pos, (insert_rule_f, new_gamma_ls))

        elif add_or_remove == add_or_remove_rule_mh_step_f.REMOVE:

            # decide remove position
            def q_remove_pos():
                if allow_end_changes:
                    try:
                        remove_pos = np.random.randint(0, theta_and_data.theta.L)
                    except:
                        print theta_and_data.theta.L
                        pdb.set_trace()
                    return remove_pos, 0.
                else:
                    if theta_and_data.theta.L == 1:
                        return 0, 0.
                    else:
                        remove_pos = np.random.randint(0, theta_and_data.theta.L-1)
                        return remove_pos, 0.

            remove_pos, remove_pos_log_q_ratio = q_remove_pos()
            log_q_ratio += remove_pos_log_q_ratio

            if np.sum(theta_and_data.theta.z_ns == remove_pos) == 0:
                debug = True
            
            def q_new_gamma_ls_simple():
                new_gamma_ls = copy.copy(theta_and_data.theta.gamma_ls)
                new_gamma_ls = np.delete(new_gamma_ls, remove_pos)
                return new_gamma_ls, 0.

            def q_new_gamma_ls_optimize_all():
                old_rule_f = theta_and_data.theta.rule_f_ls[remove_pos]
                theta_and_data.theta.rule_f_ls.pop(remove_pos)
                new_gamma_ls = theta_and_data.greedy_optimal_gamma_ls()
                theta_and_data.theta.rule_f_ls.insert(remove_pos, old_rule_f)
                return new_gamma_ls, 0

            def q_new_gamma_hybrid():
                if np.random.random() < 0.5:
#                if reduce(lambda (prev,ok), x: (x,x>=prev and ok), theta_and_data.data_p_ls, (0,True)):
                    return q_new_gamma_ls_optimize_all()
                else:
                    return q_new_gamma_ls_simple()

            q_new_gamma_ls = q_new_gamma_ls_optimize_all
            
            new_gamma_ls, new_gamma_ls_log_q_ratio = q_new_gamma_ls()
            log_q_ratio += new_gamma_ls_log_q_ratio

            if theta_and_data.theta.rule_f_idx_ls == (0,1,2,3,4) and np.random.random() < strong_debug:
                debug = True
                
            if debug:
                print '\nremove rule at', remove_pos, 'out of', theta_and_data.theta.L
                print 'BEFORE'
                print theta_and_data.informative_df()
                
            # all the decisions have been made.  now make the actual changes                
            removed_rule_f = theta_and_data.theta.rule_f_ls[remove_pos]
            old_gamma_ls = theta_and_data.theta.gamma_ls
#            removed_gamma = theta_and_data.theta.gamma_ls[remove_pos]
            theta_and_data.theta.rule_f_ls.pop(remove_pos)
            theta_and_data.theta.gamma_ls = new_gamma_ls
#            theta_and_data.theta.gamma_ls = np.delete(theta_and_data.theta.gamma_ls, remove_pos)

            # decide whether to change
            new_loglik = self.obj_f(theta_and_data)
            accept = self.accept_proposal_f(old_loglik, new_loglik)
#            new_log_theta_prob, new_log_theta_given_theta_prob = get_probs(theta_and_data)
#            log_accept_prob = log_q_ratio + (new_log_theta_prob + new_log_theta_given_theta_prob) - (old_log_theta_prob + old_log_theta_given_theta_prob)
#            accept = np.random.random() < np.exp(log_accept_prob)

            if old_rule_f_idx_ls == (0,1,2,3,4) and accept:
                theta_and_data.theta.rule_f_ls.insert(remove_pos, removed_rule_f)
                theta_and_data.theta.gamma_ls = old_gamma_ls
                print 'BEFORE LEAVING THE BEST'
                print theta_and_data.informative_df()
                theta_and_data.theta.rule_f_ls.pop(remove_pos)
                theta_and_data.theta.gamma_ls = new_gamma_ls
                debug = True
            
            if debug:
                print 'AFTER'
                print theta_and_data.informative_df()
                print 'old_loglik: %.2f new_loglik %.2f' % (old_loglik, new_loglik)
                print 'WITH OPTIMIZED GAMMAS'
                older_gamma_ls = copy.copy(theta_and_data.theta.gamma_ls)
                theta_and_data.theta.gamma_ls = theta_and_data.greedy_optimal_gamma_ls()
                print theta_and_data.informative_df()
                theta_and_data.theta.gamma_ls = older_gamma_ls
#                print 'log_q_ratio', log_q_ratio
#                print 'new_log_theta_prob', new_log_theta_prob
#                print 'old_log_theta_prob', old_log_theta_prob
#                print 'log_theta_prob_diff', new_log_theta_prob - old_log_theta_prob
#                print 'new_likelihood', new_log_theta_given_theta_prob
#                print 'old_likelihood', old_log_theta_given_theta_prob
#                print 'log_likelihood_diff', new_log_theta_given_theta_prob - old_log_theta_given_theta_prob
#                print 'accept_prob', np.exp(log_accept_prob)

            # undo the change, because this should not actual modify things
            theta_and_data.theta.rule_f_ls.insert(remove_pos, removed_rule_f)
            theta_and_data.theta.gamma_ls = old_gamma_ls
#            theta_and_data.theta.gamma_ls = np.insert(theta_and_data.theta.gamma_ls, remove_pos, removed_gamma)

            return add_or_remove_rule_mh_diff_f(add_or_remove_rule_mh_step_f.REMOVE, accept, remove_pos, (None, new_gamma_ls))

        """
        stuff below is old

            
            supp = np.sum(theta_and_data.theta.z_ns==remove_pos)
            if supp < 2:
                print 'BEFORE'
                print theta_and_data.informative_df()

            
            if debug: print 'remove pos', remove_pos
            removed_rule_f = theta_and_data.theta.rule_f_ls[remove_pos]
            removed_gamma = theta_and_data.theta.gamma_ls[remove_pos]
            if remove_pos != 0:
                theta_and_data.theta.gamma_ls[remove_pos-1] *= removed_gamma
            theta_and_data.theta.rule_f_ls.pop(remove_pos)
            theta_and_data.theta.gamma_ls = np.delete(theta_and_data.theta.gamma_ls, remove_pos)
            log_Q_prob_ratio = 0

            if reduced:
                new_log_theta_prob = self.theta_and_data_dist.theta_dist.reduced_loglik(theta_and_data.theta)
            else:
                new_log_theta_prob = self.theta_and_data_dist.theta_dist.loglik(theta_and_data.theta)
            log_theta_prob_ratio = new_log_theta_prob - old_log_theta_prob
            new_log_data_given_theta_prob = self.theta_and_data_dist.data_given_theta_dist.loglik(theta_and_data.theta, theta_and_data.data)
            log_data_given_theta_prob_ratio = new_log_data_given_theta_prob - old_log_data_given_theta_prob


            new_fake_likelihood = self.theta_and_data_dist.data_given_theta_dist.original_likelihood_dist.loglik(theta_and_data)
                    
            #log_acceptance_prob = log_Q_prob_ratio + log_theta_prob_ratio + log_data_given_theta_prob_ratio

            log_acceptance_prob = log_Q_prob_ratio + log_theta_prob_ratio + new_fake_likelihood - old_fake_likelihood

            debug2=True
            
            if supp < 2:
                print 'AFTER'
                print theta_and_data.informative_df()

                if debug2: print 'remove_pos:', remove_pos
                if debug2: print 'REMOVING'
                if debug2: print 'acceptance', log_acceptance_prob, np.exp(log_acceptance_prob)
                if debug2: print 'prior', old_log_theta_prob, new_log_theta_prob, new_log_theta_prob-old_log_theta_prob
                if debug2: print 'fake likelihoods', new_fake_likelihood, old_fake_likelihood, new_fake_likelihood-old_fake_likelihood
                if debug2: print '\n'
                
            
            if debug: print 'REMOVING'
            if debug: print 'acceptance', log_acceptance_prob, np.exp(log_acceptance_prob)
            if debug: print 'prior', old_log_theta_prob, new_log_theta_prob, new_log_theta_prob-old_log_theta_prob
            if debug: print 'fake likelihoods', new_fake_likelihood, old_fake_likelihood, new_fake_likelihood-old_fake_likelihood
            if debug: print '\n'
            
            accept = np.random.random() < np.exp(log_acceptance_prob)

            theta_and_data.theta.rule_f_ls.insert(remove_pos, removed_rule_f)
            theta_and_data.theta.gamma_ls = np.insert(theta_and_data.theta.gamma_ls, remove_pos, removed_gamma)
            
            return add_or_remove_rule_mh_diff_f(add_or_remove_rule_mh_step_f.REMOVE, accept, remove_pos)
"""
