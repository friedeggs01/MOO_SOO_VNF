from ..graph import Network, Link, Node, SFC_SET
from .ant import AntV3

import numpy as np
import copy
import os
from tqdm import tqdm
import time

class SubSwarmAntV3():
    def __init__(self, id, network: Network, sfc_set: SFC_SET, cfg):
        self.id = id
        self.network = copy.deepcopy(network)
        self.sfc_set = copy.deepcopy(sfc_set)
        self.num_sfc = len(self.sfc_set)
        self.cfg = cfg
        # self.keypoint_consume = copy.copy(self.sfc_set.keypoint_consume)
        self.total_installed_vnf = 0
        self.fitness = -1
        self.finished = False

    def run(self):
        # set up parameters
        self.Q = self.cfg.Q

        # init ant family
        self.ants = []
        for i, sfc in enumerate(self.sfc_set.sfc_set):
            self.ants.append(AntV3(i, self.network, sfc, self.cfg))

        # process
        while(True):
            for ant in self.ants:
                ant.run()
        
            # if there are ants which dont satisfy the requests
            flag = False
            for ant in self.ants:
                if ant.finished is False: flag = True
            if flag: break

            # check the stopping criteria
            for sfc in self.sfc_set.sfc_set:
                if sfc.finished is False: break
            else:
                self.finished = True 
                break

        self.compute_fitness()
        self.count_actived_servers()

    def compute_fitness(self):
        if self.finished is False: # penalty
            self.R_cap, self.R_server, self.R_vnf = 1, 1, 1
        else:
            self.R_cap = (np.sum(np.array([link.total_delay for link in self.network.L.values()])) + np.sum(np.array([server.total_delay for server in self.network.N_server]))) / ((self.network.total_delay_link + self.network.total_delay_server) * self.num_sfc)
            self.R_server = np.sum(np.array([self.network.N[id].cost for id in self.network.server_ids if len(self.network.N[id].vnf_used) > 0])) / self.network.sum_cost_servers
            self.R_vnf = np.sum(np.array([self.network.N[id].total_installed_vnf_cost for id in self.network.server_ids])) / np.sum(self.network.cost_vnfs)
        self.objective_value = self.R_cap + self.R_server + self.R_vnf
        self.fitness = self.Q / self.objective_value 

    def count_actived_servers(self):
        self.num_actived_servers = self.network.count_actived_servers() if self.finished else -1

    def __repr__(self) -> str:
        return f"Subswarm ant {self.id} | Fitness: {self.fitness} | Obj: {self.objective_value}({self.R_cap},{self.R_server},{self.R_vnf})"
    
    def __str__(self) -> str:
        return self.__repr__()