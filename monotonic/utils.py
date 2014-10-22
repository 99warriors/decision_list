import monotonic.monotonic.constants as constants
import numpy as np
import scipy.stats
import copy
import python_utils.python_utils.caching as caching

class my_object(object):

    @classmethod
    def get_cls(cls):
        return cls
    
    def __getitem__(self, s):
        try:
            return self.__dict__[s]
        except KeyError:
            return cls.__dict__[s].__get__(self)

    def __setitem__(self, s, val):
        try:
            self.__dict__[s] = val
        except KeyError:
            cls.__dict__[s].__set__(self, val)

obj_base = my_object
f_base = my_object

import cProfile

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

class datum(obj_base):

    def __init__(self, id, x, y):
        self.id, self.x, self.y = id, x, y
        
class data(obj_base):
    """
    
    """
    def __len__(self):
        return len(self.x_ns)
    
    def __init__(self, data_id, id_ns, x_ns, y_ns, x_names = None):
        if y_ns is None:
            y_ns = [None for i in xrange(len(id_ns))]
        self.id_ns, self.x_ns, self.y_ns = id_ns, np.array(x_ns), np.array(y_ns)
        self.datums = [datum(i,x,y) for (i,x,y) in zip(id_ns, x_ns,y_ns)]
        import python_utils.python_utils.caching as caching
        #self.my_hash = hash(caching.get_hash(self))
        self.data_id = data_id
        self.x_names = x_names

    def __iter__(self):
        return iter(self.datums)
        
    def __getitem__(self, i):
        return self.datums[i]

    def get_Xy(self):
        return self.x_ns, self.y_ns

    def pretty_print(self):
        import string
        return string.replace(str(self.data_id), ' ', '')
    
    def __hash__(self):
        return hash(self.data_id)

class iden(f_base):

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val

def raw_data_to_monotonic_input(d):
    """
    assume label is in last column of dataframe
    this is only for real data, because i am assigning the id based on hash.  in simulation study the data are identical over runs, but are different runs.
    """
    ys = list(d.iloc[:,-1])
    xs = [tuple(row) for (row_name, row) in d.iloc[:,0:-1].iterrows()]
    data_id = hash(caching.get_hash((xs,ys)))
    import pdb
#    xs = [tuple(row) for (row_name, row) in d[[col for col in d.columns if col != y_name]].iterrows()]

    return data(data_id, range(len(ys)), xs, ys, d.columns[0:-1])
    
no_ys_data = data
        
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(x):
    try:
        return np.log(x) - np.log(1-x)
    except:
        import pdb
        pdb.set_trace()

def reverse_np_array(ar):
    return np.array([x for x in reversed(ar)])

def vectorized_zero_truncated_poisson_sample(rates):
    ts = -np.log(1.0 - (np.random.uniform(size=len(rates)) * (1.0 - np.exp(-1.0*rates))))
    new_rates = rates - ts
    return 1.0 + scipy.stats.poisson(mu=new_rates).rvs()

def sample_truncated_gamma(alpha, beta, left_truncate):
    d = scipy.stats.gamma(a=alpha, scale=1./beta)
    omitted = d.cdf(left_truncate)
    u = np.random.sample()
    
    ans = d.ppf(u * (1.-omitted) + omitted)
    eps = .01
    if not np.isfinite(ans):
        import pdb
#        pdb.set_trace()
        return left_truncate + eps
    else:
        return ans

def sample_zero_truncated_negative_binomial(success_probs, num_successes):
    """
    returns number of failures required to get num_successes
    """
    # if number of failures required to > 0, figure out num_failures F1 to get first success, aka negbin(p,1) = geo(p)
    # or figure out number of successes received until get first failure.  Then still need num_successes - 1.  number of failures until getting it is negbin(p, num_successes-1)
    # but this above sampling step might say that number of successes received until first failure is greater than num_successes.  ok give up on principled sampling
    import pdb
 
    d = scipy.stats.nbinom(n=num_successes,p=success_probs)
    assert len(success_probs) == len(num_successes)
    zero_probs = d.pmf(np.zeros(len(success_probs)))
    slicers = np.random.uniform(low=zero_probs, high=np.ones(len(success_probs)))
#    slicers = scipy.stats.uniform(loc=zero_probs,scale=np.ones(len(success_probs))-zero_probs).rvs()
    return d.ppf(slicers)
    
def swap_list_items(l, idx_a, idx_b):
    a_handle = l[idx_a]
    b_handle = l[idx_b]
    l[idx_b] = a_handle
    l[idx_a] = b_handle
        
class my_list(list, obj_base):
    """
    set of objects that can be inserted/removed is pre-set
    """
    def sample(self):
        """
        sample without replacement
        """
        pass

    def insert(self, i, elt):
        pass

    def remove(self, i):
        pass

class theta_copier_f(f_base):

    def __init__(self, params_to_copy):
        self.params_to_copy = params_to_copy

    def __call__(self, theta):
        new_theta = copy.copy(theta)
        for param in self.params_to_copy:
            new_theta[param] = copy.deepcopy(theta[param])
        return new_theta

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
 
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
 
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]
