import numpy as np
import monotonic.monotonic.constants as constants
import monotonic.monotonic.utils as monotonic_utils
import copy
import monotonic.monotonic.model as model

class trace(monotonic_utils.obj_base):

    def __init__(self, start_theta, diff_fs, theta_copier_f):
        self.start_theta, self.diff_fs, self.theta_copier_f = start_theta, diff_fs, theta_copier_f

    def get_proposal_thetas(self, whether_to_log_f):
        pass
        
    def get_thetas(self, whether_to_log_f):
        assert self.start_theta.L+1 == len(self.start_theta.gamma_ls)
        thetas = [copy.deepcopy(self.start_theta)]
        cur_theta = copy.deepcopy(self.start_theta)
        debug = False
        for diff_f in self.diff_fs:
            diff_f(cur_theta)
            if debug:
                print diff_f.param_idx
                print len(cur_theta.gamma_ls), cur_theta.L+1
                print cur_theta.gamma_ls
                print cur_theta.rule_f_ls
            assert len(cur_theta.gamma_ls) == cur_theta.L+1
            if whether_to_log_f(diff_f):
#                new_theta = copy.deepcopy(cur_theta)
                new_theta = self.theta_copier_f(cur_theta)
                thetas.append(new_theta)
        return thetas
#                theta_and_datas.append(copy.deepcopy(cur_theta_and_data))
#        return theta_and_datas

                
