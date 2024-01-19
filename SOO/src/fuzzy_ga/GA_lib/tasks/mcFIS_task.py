import numpy as np
from .task import AbstractTask
from ...mcFIS import mcFIS

class mcFISTask(AbstractTask):
    def __init__(self, networks, sfc_sets, cfg, dimA=5, dimB=5):
        self.networks = networks
        self.sfc_sets = sfc_sets
        self.cfg = cfg
        self.dimA = dimA
        self.dimB = dimB
        self.dim = self.dimA + self.dimB

    # calculate fitness        
    def __call__(self, x):
        A, B = x[:self.dimA], x[self.dimA:]
        self.mcfis_list = [mcFIS(network, sfc_set, self.cfg) for network, sfc_set in zip(self.networks, self.sfc_sets)]
        for mcfis in self.mcfis_list:
            mcfis.run(A, B)
        
        self.objective_list = [mcfis.objective for mcfis in self.mcfis_list]
        self.fitness = 1 / np.mean(self.objective_list)
        return self.fitness, self.objective_list
    
    # initial randomly
    def generate_gene(self):
        A = np.random.uniform(low=0, high=1, size=5)
        while(np.sum(A) == 0):
            A = np.random.uniform(low=0, high=1, size=5)
        B = np.random.choice(a=[1,2,3], size=5)
        return np.concatenate([A,B])
