import numpy as np
import functools
import scipy.stats
import monotonic.monotonic.utils as monotonic_utils
import monotonic.monotonic.distributions as distributions
import itertools
import pdb
import python_utils.python_utils.caching as caching

##### representation of a decision list #####

class reduced_theta(monotonic_utils.obj_base):
    """
    parameters of model, represents a decision list in that it can be used to predict p(y|x)
    length of rule_f_ls should be 1 less than that of gamma_ls
    """
    @property
    def L(self):
        return len(self.rule_f_ls)
    
    @property
    def r_ls(self):
        return np.log(self.v_ls)

    @property
    def p_ls(self):
        return np.array(map(monotonic_utils.logistic, self.r_ls))
        
#    @property
#    def p_ls(self):
#        return map(monotonic_utils.logistic, self.r_ls)


    
    @property
    def v_ls(self):
        def reverse(l):
            return [x for x in reversed(l)]
        import pdb
        try:
            ans = np.exp(np.array(reverse(np.cumsum(reverse(np.log(self.gamma_ls))))))
        except:
            pdb.set_trace()

        debug = False
        if debug:
            try:
                for i in range(len(ans)-1):
                
                    assert ans[i] > ans[i+1]
            except:
                print 'v_ls', ans
                print 'gamma_ls', self.gamma_ls
                print reverse(self.gamma_ls)
                print np.cumsum(reverse(self.gamma_ls))
                pdb.set_trace()
        return ans

    def get_z_n(self, datum):
        ans = np.argmax(map(lambda rule_f: rule_f(datum), self.rule_f_ls) + [True])
        return ans

    def by_z_ns(self, var_ns):
        by_z_ns = [[] for i in xrange(self.L+1)]
        for (var_n, z_n) in zip(var_ns, self.z_ns):
            by_z_ns[z_n].append(var_n)
        return by_z_ns
    
    @property
    def rule_f_idx_ls(self):
        return tuple([rule_f_l.idx for rule_f_l in self.rule_f_ls])
    
    def __init__(self, rule_f_ls, gamma_ls):
        self.rule_f_ls, self.gamma_ls = rule_f_ls, gamma_ls

class theta(reduced_theta):

    def __init__(self, rule_f_ls, gamma_ls, w_ns, zeta_ns, data):
        self.w_ns, self.zeta_ns, self.data = w_ns, zeta_ns, data
        reduced_theta.__init__(self, rule_f_ls, gamma_ls)

    @property
    def support_ls(self):
        return np.array(map(lambda i: np.sum(self.data.y_ns[self.z_ns==i]), xrange(0,self.L+1)))
        
    @property
    def z_ns(self):
        return self.z_ns_helper_faster(self.rule_f_idx_ls)
#        return self.z_ns_helper_faster(self.rule_f_idx_ls)

    @caching.hash_cache_method_decorator
    def z_ns_helper(self, rule_f_idx_ls):
        return np.array(map(self.get_z_n, self.data.datums))

    @caching.hash_cache_method_decorator
    def z_ns_helper_faster(self, rule_f_idx_ls):
        # make a num_rules x N matrix.  each rule contributes a row
        return np.argmax(np.array([rule_f_l.batch_call(self.data) for rule_f_l in self.rule_f_ls]+[np.ones(len(self.data),dtype=bool)]), axis=0)
    
    @property
    def p_ns(self):
        try:
            return self.p_ls[self.z_ns]
        except:
            pdb.set_trace()

    @property
    def v_ns(self):
        return self.v_ls[self.z_ns]

    @property
    def N(self):
        return len(self.data)
    
##### prior over z_ns #####

class zeta_ns_dist(distributions.vectorized_dist):

    def __init__(self, N, rate):
        self.N, self.rate = N, rate
    
    def batch_loglik(self, zeta_ns):
        return scipy.stats.expon(scale=self.rate).logpdf(zeta_ns)

    def batch_sample(self):
        return scipy.stats.expon(scale=self.rate).rvs(self.N)

##### prior over w_ns #####
        
class w_ns_given_zeta_ns_given_v_ns_dist(distributions.conditional_vectorized_dist):

    def batch_loglik(self, (zeta_ns, v_ns), w_ns):
        return np.sum(scipy.stats.poisson(mu=zeta_ns*v_ns).logpmf(w_ns))
    
    def batch_sample(self, (zeta_ns, v_ns)):
        return scipy.stats.poisson(mu=zeta_ns*v_ns).rvs()        

##### prior over gamma_ls given L #####

class gamma_ls_given_L_dist(distributions.conditional_vectorized_dist):

    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
        self.horse = distributions.gamma_dist(alpha, beta)
        
    def batch_loglik(self, L, gamma_ls):
        return 0
        import pdb
        try:
            return scipy.stats.gamma(a=self.alpha*np.ones(L+1), scale=1./(self.beta*np.ones(L+1))).logpdf(gamma_ls)
        except:
            pdb.set_trace()

    def batch_sample(self, L):
        #raise NotImplementedError
        return scipy.stats.gamma(a=self.alpha*np.ones(L+1), scale=1./(self.beta*np.ones(L+1))).rvs()

    def iterative_sample(self, sampled_gammas):
        return monotonic_utils.sample_truncated_gamma(self.alpha, self.beta, 1.)
        return self.horse.sample()
    
    def get(self, L, i):
        return self.horse
    
##### prior over decision lists #####
            
class theta_dist(distributions.dist):

    def __init__(self, rule_f_ls_given_L_dist, gamma_ls_given_L_dist, zeta_ns_dist, L_dist, x_ns_data):
        self.rule_f_ls_given_L_dist, self.gamma_ls_given_L_dist, self.zeta_ns_dist, self.L_dist, self.x_ns_data = rule_f_ls_given_L_dist, gamma_ls_given_L_dist, zeta_ns_dist, L_dist, x_ns_data
        self.w_ns_given_zeta_ns_given_v_ns_dist = w_ns_given_zeta_ns_given_v_ns_dist()

    def reduced_loglik(self, theta):
        log_p = 0.0
        log_p += self.L_dist.loglik(theta.L)
        log_p += np.sum(self.gamma_ls_given_L_dist.batch_loglik(theta.L,theta.gamma_ls))
        log_p += self.rule_f_ls_given_L_dist.loglik(theta.L, theta.rule_f_ls)
#        print log_p, 'reduced_theta'
        return log_p

    def sample(self):
        L = self.L_dist.sample() + 1
        rule_f_ls = self.rule_f_ls_given_L_dist.sample(L)
        gamma_ls = self.gamma_ls_given_L_dist.batch_sample(L)
        zeta_ns = self.zeta_ns_dist.batch_sample()
        theta_sample = theta(rule_f_ls, gamma_ls, None, zeta_ns, self.x_ns_data)
        theta_sample.w_ns = self.w_ns_given_zeta_ns_given_v_ns_dist.batch_sample((theta_sample.zeta_ns, theta_sample.v_ns))
        return theta_sample
        
    def loglik(self, theta):
        log_p = 0.0
        log_p += self.L_dist.loglik(theta.L)
        debug = False
        if debug:
            print len(theta.gamma_ls), theta.L+1
            print theta.gamma_ls
            print theta.rule_f_ls
        assert len(theta.gamma_ls) == theta.L+1
        log_p += np.sum(self.gamma_ls_given_L_dist.batch_loglik(theta.L,theta.gamma_ls))
        log_p += np.sum(self.zeta_ns_dist.batch_loglik(theta.zeta_ns))
        log_p += np.sum(self.w_ns_given_zeta_ns_given_v_ns_dist.batch_loglik((theta.zeta_ns, theta.v_ns), theta.w_ns))
        log_p += self.rule_f_ls_given_L_dist.loglik(theta.L, theta.rule_f_ls)
        return log_p

class theta_dist_constructor(monotonic_utils.f_base):

    def __init__(self, rule_f_ls_given_L_dist_constructor, gamma_ls_given_L_dist, L_dist):
        self.rule_f_ls_given_L_dist_constructor, self.gamma_ls_given_L_dist, self.L_dist = rule_f_ls_given_L_dist_constructor, gamma_ls_given_L_dist, L_dist

    def __call__(self, data):
        rule_f_ls_given_L_dist = self.rule_f_ls_given_L_dist_constructor(data)
        return theta_dist(rule_f_ls_given_L_dist, self.gamma_ls_given_L_dist, zeta_ns_dist(len(data), 1.0), self.L_dist, data)

        
##### data distribution #####
        
class data_given_theta_dist(distributions.conditional_vectorized_dist):
    """
    returns a data (bunch of datum)
    """

    def batch_sample(self, theta):
        return monotonic_utils.data(theta.data.data_id, theta.data.id_ns, theta.data.x_ns, np.array(map(int, np.array(theta.w_ns) != 0 )))

    def loglik(self, theta, data):
        if np.array_equal(theta.w_ns > 0, data.y_ns == 1):
            return 0
        else:
            assert False

##### joint distribution over theta and data #####

class theta_and_data(monotonic_utils.obj_base):

    def __init__(self, theta, data):
        self.theta, self.data = theta, data

    @property
    def z_ns(self):
        return self.z_ns_helper(self.rule_f_idx_ls)

#    @caching.hash_cache_method_decorator
#    def data_by_z_ns_helper(self, rule_f_idx_ls):
#        return np.array(map(self.get_z_n, self.data.datums))
        
    @property
    def data_p_ls(self):
        def temper(y):
#            return y
            ans = max(min(y,.99), .01)
            return ans
        return np.array(map(lambda i: temper(np.mean(self.data.y_ns[self.theta.z_ns==i]) if np.sum(self.theta.z_ns==i) > 0 else 0.5), xrange(0,self.theta.L+1)))


    
#    @property
    def data_by_z_n(self):
        return self.data_by_z_ns_helper(self.theta.rule_f_idx_ls)

    def greedy_optimal_gamma_ls(self):
        data_r_ls = map(monotonic_utils.logit, reversed(self.data_p_ls))
#        print data_r_ls
        monotonic_r_ls = np.array(reduce(lambda accum, x: accum+[max(accum+[x])], data_r_ls,[]))
#        print monotonic_r_ls
        monotonic_v_ls = np.exp(monotonic_r_ls)
        ans = monotonic_utils.reverse_np_array(np.exp(map(lambda l:monotonic_r_ls[l]-monotonic_r_ls[l-1] if l > 0 else monotonic_r_ls[l], range(self.theta.L+1))))
        for x in ans:
            if not np.isfinite(x):
                print ans
                pdb.set_trace()
#        print ans
        return ans
        
        
    @caching.hash_cache_method_decorator
    def data_by_z_ns_helper(self, rule_f_idx_ls):
        id_l = [[] for i in xrange(self.theta.L+1)]
        x_l = [[] for i in xrange(self.theta.L+1)]
        y_l = [[] for i in xrange(self.theta.L+1)]
        for datum, z_n in itertools.izip(self.data, self.theta.z_ns):
            id_l[z_n].append(datum.id)
            x_l[z_n].append(datum.x)
            y_l[z_n].append(datum.y)
        asdf = 0
        return [monotonic_utils.data(asdf, ids, xs, ys) for (ids, xs, ys) in itertools.izip(id_l, x_l, y_l)]

    def informative_df(self):
        by = self.data_by_z_n()
        ok = []
        import pandas as pd
        total_prob = 0.0
        for (rule_f, d, gamma_l, p_l) in zip(self.theta.rule_f_ls + [None], by, self.theta.gamma_ls, self.theta.p_ls):
            try:
                pos = sum(d.y_ns)

                asdf = rule_f.idx
            except:
                asdf = 'default'
            logprob = np.sum(np.log(p_l) * d.y_ns + np.log(1-p_l) * (1.-d.y_ns))
            total_prob += logprob
            support = rule_f.support if not rule_f is None else np.nan
            rep = repr(rule_f) if not rule_f is None else np.nan
            try:
                ok.append(pd.Series({'rule':asdf, 'pos':pos/float(len(d)), 'supp':len(d), 'p':p_l, 'gamma':gamma_l, 'prob':logprob, 'supp_overall':support, 'z':rep}))
            except ZeroDivisionError:
                print 'zero'
                ok.append(pd.Series({'rule':asdf, 'pos':0., 'supp':len(d), 'p':p_l, 'gamma':gamma_l, 'prob':logprob, 'supp_overall':support, 'z':rep}))

        ok.append(pd.Series({'prob':total_prob}))
        return pd.DataFrame(ok)

class theta_and_data_dist(distributions.dist):

    def __init__(self, theta_dist, data_given_theta_dist):
        self.theta_dist, self.data_given_theta_dist = theta_dist, data_given_theta_dist

    def loglik(self, theta_and_data):
        return self.theta_dist.loglik(theta_and_data.theta) + self.data_given_theta_dist.loglik(theta_and_data.theta, theta_and_data.data)

    def sample(self):
        sampled_theta = self.theta_dist.sample()
        sampled_data = self.data_given_theta_dist.batch_sample(sampled_theta)
        return theta_and_data(sampled_theta, sampled_data)

    def reduced_theta_loglik(self, theta_and_data):
        return self.theta_dist.reduced_loglik(theta_and_data.theta)
    
    def reduced_data_given_theta_loglik(self, theta_and_data):
        try:
            return np.sum(np.log(theta_and_data.theta.p_ns) * theta_and_data.data.y_ns + np.log(1.-theta_and_data.theta.p_ns) * (1.-theta_and_data.data.y_ns))
        except:
            pdb.set_trace()

    def reduced_loglik(self, theta_and_data):
        return self.reduced_theta_loglik(theta_and_data) + self.reduced_data_given_theta_loglik(theta_and_data)
    
##### some theta distributions with various parameters fixed #####

class theta_fixed_rule_f_ls_fixed_gamma_ls_dist(distributions.conditional_dist):

    def __init__(self, rule_f_ls, gamma_ls, zeta_ns_dist, x_ns_data):
        self.rule_f_ls, self.gamma_ls, self.zeta_ns_dist, self.x_ns_data = rule_f_ls, gamma_ls, zeta_ns_dist, x_ns_data
        self.L = len(rule_f_ls)
        self.w_ns_given_zeta_ns_given_v_ns_dist = w_ns_given_zeta_ns_given_v_ns_dist()
                
    def sample(self):
        theta_sample = theta(self.rule_f_ls, self.gamma_ls, None, None, self.x_ns_data)
        theta_sample.zeta_ns = self.zeta_ns_dist.batch_sample()
        theta_sample.w_ns = self.w_ns_given_zeta_ns_given_v_ns_dist.batch_sample((theta_sample.zeta_ns, theta_sample.v_ns))
        return theta_sample

    def loglik(self, theta):
        log_p = 0.0
        log_p += np.sum(self.zeta_ns_dist.batch_loglik(theta.z_ns))
        log_p += np.sum(self.w_ns_given_zeta_ns_given_v_ns_dist.batch_loglik((theta.zeta_ns, v_ns), theta.w_ns))
        return log_p

class theta_fixed_rule_f_ls_dist(distributions.dist):

    def __init__(self, rule_f_ls, gamma_ls_given_L_dist, zeta_ns_dist, x_ns_data):
        self.rule_f_ls, self.gamma_ls_given_L_dist, self.zeta_ns_dist, self.x_ns_data = rule_f_ls, gamma_ls_given_L_dist, zeta_ns_dist, x_ns_data
        self.L = len(rule_f_ls)
        self.w_ns_given_zeta_ns_given_v_ns_dist = w_ns_given_zeta_ns_given_v_ns_dist()
        
    def sample(self):
        theta_sample = theta(self.rule_f_ls, None, None, None, self.x_ns_data)
        theta_sample.gamma_ls = self.gamma_ls_given_L_dist.batch_sample(self.L, l)
        theta.zeta_ns = self.zeta_ns_dist.batch_sample()
        theta_sample.w_ns = self.w_ns_given_zeta_ns_given_v_ns_dist.batch_sample((theta_sample.zeta_ns, theta_sample.v_ns))
        return theta_sample

    def loglik(self, theta):
        log_p = 0.0
        log_p += np.sum(self.gamma_ls_given_L_dist.batch_loglik(theta.L, theta.gamma_ls))
        log_p += np.sum(self.zeta_ns_dist.batch_loglik(theta.z_ns))
        log_p += np.sum(self.w_ns_given_zeta_ns_given_v_ns_dist.batch_loglik((theta.zeta_ns, theta.v_ns), theta.w_ns))
        return log_p

##### some priors over the rules #####

class agnostic_rule_f_ls_only_order_unknown_dist(distributions.conditional_dist):
    """
    all orderings have the same probability
    """
    def __init__(self, fixed_rule_f_ls):
        self.fixed_rule_f_ls = set(fixed_rule_f_ls)
    
    def loglik(self, L, rule_f_ls):
        assert np.all([rule_f_l in self.fixed_rule_f_ls for rule_f_l in rule_f_ls])
        assert L == len(self.fixed_rule_f_ls)
        return -np.log(float(len(self.fixed_rule_f_ls)))

class uniform_base_rule_dist(distributions.dist):

    def __init__(self, possible_rule_fs):
        self.possible_rule_fs = possible_rule_fs

    def loglik(self, rule_f):
        return -np.log(len(self.possible_rule_fs))

    def sample(self):
        return np.random.choice(self.possible_rule_fs)
    
class with_replacement_rule_f_ls_given_L_dist(distributions.conditional_dist):

    def __init__(self, base_rule_dist):
        self.base_rule_dist = base_rule_dist

    def loglik(self, L, rule_f_ls):
        return np.sum([self.base_rule_dist.loglik(rule_f_l) for rule_f_l in rule_f_ls])

    def sample(self, L):
        return reduce(lambda sampled, dummy: sampled + [self.iterative_sample(sampled)], [])

    def iterative_sample(self, sampled_rule_fs):
        return self.base_rule_dist.sample()

class fixed_set_rule_f_ls_given_L_dist(distributions.conditional_dist):

    def __init__(self, possible_rule_fs):
        self.possible_rule_fs = set(possible_rule_fs)
        self.probs = -1.*np.cumsum(np.log(np.arange(len(self.possible_rule_fs), 0, -1)))
        self.possible_rule_fs_list = list(possible_rule_fs)
                    
    def loglik(self, L, rule_f_ls):
        return self.probs[len(rule_f_ls)-1]

    def iterative_sample(self, sampled_rule_fs):
        asdf = set(sampled_rule_fs)
        for i in np.random.permutation(len(self.possible_rule_fs_list)):
            if self.possible_rule_fs_list[i] not in asdf:
                return self.possible_rule_fs_list[i]
        assert False
        return np.random.choice(tuple(self.possible_rule_fs - set(sampled_rule_fs)))

    def iterative_sample_loglik(self, sampled_rule_fs, iterative_sample):
        return -np.log(len(self.possible_rule_fs) - len(sampled_rule_fs))

    def sample(self, L):
        while 1:
            ans = list(np.random.choice(self.possible_rule_fs_list, L))
            if len(ans) > 0:
                return ans
    
class fixed_set_rule_f_ls_given_L_dist_constructor(monotonic_utils.f_base):

    def __init__(self, rule_miner_f):
        self.rule_miner_f = rule_miner_f

    def __call__(self, data):
        rule_fs = self.rule_miner_f(data)
        return fixed_set_rule_f_ls_given_L_dist(rule_fs)
        
##### original likelihood y_ns | gamma_ls with w_ns and zeta_ns marginalized out #####

class original_likelihood_dist(distributions.dist):

    def loglik(self, theta_and_data):
        return np.sum(np.log(theta_and_data.theta.p_ns) * theta_and_data.data.y_ns + np.log(1.-theta_and_data.theta.p_ns) * (1.-theta_and_data.data.y_ns))
