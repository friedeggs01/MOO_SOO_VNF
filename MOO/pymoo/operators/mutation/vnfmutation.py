import numpy as np
import copy
import random

from pymoo.core.mutation import Mutation

class VNFmutation(Mutation):
    def __init(self, prob=0.9, **kwargs):
        super().__init__(**kwargs)

    def _do(self, problem, X, params=None, **kwargs):
        num_server = problem.num_server
        vnf_max = problem.vnf_max
        n_matings, _ = X.shape
        Xp = copy.deepcopy(X)
        
        for j in range(n_matings):

            ser1 = random.randint(0, num_server-1)
            ser2 = random.randint(0, num_server-1)
            while ser2 == ser1:
                ser2 = random.randint(0, num_server-1)

            for i in range(vnf_max):
                temp = Xp[j][ser1*vnf_max+i]
                Xp[j][ser1*vnf_max+i] = Xp[j][ser2*vnf_max+i]
                Xp[j][ser2*vnf_max+i] = temp
                
        return Xp