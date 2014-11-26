import pdb
import numpy as np
import fim
import monotonic.monotonic.utils as monotonic_utils
import pandas as pd
import extra_utils as caching

class rule_f(monotonic_utils.f_base):
    """
    just a boolean fxn on x
    """
    def __init__(self, _idx, r, support = None, x_names = None):
        self._idx, self.r, self.support, self.x_names = _idx, r, support, x_names
        self.cardinality = len(r)

    def __repr__(self):
        if self.x_names != None:
            return str(self.x_names)
        else:
            return str(self.r)
        
    @property
    def idx(self):
        return self._idx

    def __hash__(self):
        return hash(self.idx)
    
    def __call__(self, datum):
        import pdb
        for r_i in self.r:
            if not datum.x[r_i]:
                return False
        return True

    @caching.hash_cache_method_decorator
    def batch_call(self, data):
        return np.array([self(datum) for datum in data])

#@caching.default_cache_fxn_decorator
#@caching.default_read_fxn_decorator
#@caching.default_write_fxn_decorator
class rule_miner_f(monotonic_utils.f_base):
    """
    supp is the proportion of data that needs to satisfy the rule
    zmin is the cardinality
    """
    def __init__(self, supp, zmax):
        self.supp, self.zmax = supp, zmax
    
    def __call__(self, data):
        def which_are_1(v):
            return list(pd.Series(range(len(v)))[map(bool,v)])
        length = float(len(data))
        import pdb
        raw = fim.fpgrowth([which_are_1(x_n) for x_n in data.x_ns], supp = self.supp, zmax = self.zmax)
        data_idx = hash(data)
#        for (r,s) in raw:
#            try:
#                print data.x_names[r]
#            except:
#                pdb.set_trace()


        if data.x_names != None:
            return [rule_f((data_idx,i), r, s[0]/length, list(data.x_names[list(r)])) for (i, (r, s)) in enumerate(raw)]
        else:
            return [rule_f((data_idx,i), r, s[0]/length) for (i, (r, s)) in enumerate(raw)]
        
class hard_coded_rule_f(monotonic_utils.f_base):
    """
    returns True on hardcoded set of data.  truth should be boolean vector
    """
    def __init__(self, _idx, truth):
        self._idx, self.truth = _idx, truth
        self.support = np.sum(self.truth) / len(self.truth)

    def __repr__(self):
        return repr(self.idx)
        
    def __hash__(self):
        return hash(self.idx)

    def __eq__(self, other):
        return self.idx == other.idx
    
    @property
    def idx(self):
        return self._idx
        
    def __call__(self, datum):
        return self.truth[datum.id]

    @caching.hash_cache_method_decorator
    def batch_call(self, data):
        return self.truth[np.array([datum.id for datum in data])]
    
class get_hard_coded_rule_f(monotonic_utils.f_base):

    def __init__(self, p_true, num_rules):
        self.p_true, self.num_rules = p_true, num_rules

    def __call__(self, data):
        import python_utils.python_utils.caching as caching
        import pdb
        data_hash = hash(data)
        return [hard_coded_rule_f((data_hash,j), np.array([np.random.random() < self.p_true for i in xrange(len(data))])) for j in xrange(self.num_rules)]
