import numpy as np
import monotonic.monotonic.constants as constants
from collections import namedtuple
import monotonic.monotonic.utils as monotonic_utils
import monotonic.monotonic.trace as trace
import copy
import monotonic.monotonic.model as model

class diff_f(monotonic_utils.f_base):
    """
    represents the change to old theta proposed by a mcmc step
    changes theta_and_data in place
    """
    @property
    def accepted(self):
        pass

    def make_change(self, theta):
        pass

    def __call__(self, theta):
        if self.accepted:
            self.make_change(theta)
            supports = theta.support_ls
            for l in range(theta.L-1,-1,-1):
                if supports[l] == 0:
                    theta.gamma_ls = np.delete(theta.gamma_ls, l)
                    theta.rule_f_ls.pop(l)
    
class mcmc_step_f(monotonic_utils.f_base):

    def __init__(self, p_theta):
        pass
    
    def __call__(self, theta):
        """
        should not modify theta, at least in the end
        """
        pass
    
class run_mcmc_f(monotonic_utils.f_base):
    """
    theta is modified in place, but full history can be recovered using the diff_fs
    for now, just cycle between the different updates
    agrees to not modify the data in theta_and_data, since it's being viewed as fixed
    """
    def __call__(self, mcmc_step_fs, start_theta_and_data, num_steps, theta_copier_f):
        theta_and_data = copy.deepcopy(start_theta_and_data)
        diff_fs = []
        for i in xrange(num_steps):
            for mcmc_step_f in mcmc_step_fs:
                diff_f = mcmc_step_f(theta_and_data)
                debug = False
                if debug:
                    print diff_f.param_idx
                    print len(theta_and_data.theta.gamma_ls), theta_and_data.theta.L+1
                    print theta_and_data.theta.gamma_ls
                    print theta_and_data.theta.rule_f_ls
#                if diff_f.accepted:
                diff_f(theta_and_data.theta)
                diff_fs.append(diff_f)
            if i % 500 == 0:

                print 'STEP',i,''
#                print theta_and_data.informative_df()
                if False:
                    print 'step', i, theta_and_data.theta.rule_f_idx_ls
                    print 'gamma_ls', theta_and_data.theta.gamma_ls
                    print 'p_ls', theta_and_data.theta.p_ls
                    print 'actual p_ls', theta_and_data.data_p_ls
                    import pandas as pd
                    print 'num at nodes', np.array(pd.Series(theta_and_data.theta.z_ns).value_counts())
        return trace.trace(start_theta_and_data.theta, diff_fs, theta_copier_f)

class get_traces_f(monotonic_utils.f_base):

    def __init__(self, theta_dist_constructor, mcmc_step_f_constructors, num_steps):
        params_to_copy = ['rule_f_ls', 'gamma_ls']
        self.theta_copier_f = monotonic_utils.theta_copier_f(params_to_copy)
        self.theta_dist_constructor, self.mcmc_step_f_constructors, self.num_steps = theta_dist_constructor, mcmc_step_f_constructors, num_steps
        
    def __call__(self, data):
        theta_dist = self.theta_dist_constructor(data)
        theta_and_data_dist = model.theta_and_data_dist(theta_dist, model.data_given_theta_dist())
        mcmc_step_fs = [mcmc_step_f_constructor(theta_and_data_dist) for mcmc_step_f_constructor in self.mcmc_step_f_constructors]
        start_theta = theta_dist.sample()
        import pdb
#        pdb.set_trace()
        start_theta_and_data = model.theta_and_data(start_theta, data)
        traces = run_mcmc_f()(mcmc_step_fs, start_theta_and_data, self.num_steps, self.theta_copier_f)
        return traces
