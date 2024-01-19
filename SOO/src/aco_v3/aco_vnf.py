from ..graph import Network, Link, Node, SFC_SET
from .ant import AntV3
from .subswarm_ant import SubSwarmAntV3
from utils import next_path

import numpy as np
import random
import copy
import json
import os
from tqdm import tqdm
import time

class ACO_VNF_V3():
    def __init__(self, network: Network, sfc_set: SFC_SET):
        self.network = copy.deepcopy(network)
        self.sfc_set = copy.deepcopy(sfc_set)
        self.keypoint_consume = self.sfc_set.keypoint_consume
        self.solution = []
        self.swarm = None
        self.total_time = None
        self.objective_value = 3
        self.num_actived_servers = None       

    def run(self, cfg):
        begin = time.time()

        if cfg.log_path:
            cfg.log_path = next_path(cfg.log_path+"%s")
            os.makedirs(cfg.log_path, exist_ok=True)

        # save config
        if cfg.log_path:
            opt = dict()
            for k,v in cfg.__class__.__dict__.copy().items():
                if "__" not in k: opt[k] = v
            json.dump(opt, open(f"{cfg.log_path}/opt.json", "w"), indent=2)

        set_seed(cfg.seed)

        # update prior info
        # for sfc in self.sfc_set.sfc_set:
        #     self.keypoint_consume[sfc.source] -= sfc.memory
            
        # aco algorithm
        # init pheromone
        if cfg.use_sfc_pheromone: 
            self.network.build_pheromone(len(self.sfc_set))
        else:
            self.network.build_pheromone()
        global_best_subswarm = None
        arg_global_best = 0
        pbar = tqdm(range(cfg.num_epochs)) if cfg.display else range(cfg.num_epochs)
        for iter_loop in pbar:
            self.swarm = [SubSwarmAntV3(i, self.network, self.sfc_set, cfg) for i in range(cfg.num_families)]
            for subswarm in self.swarm:
                subswarm.run()

            # ranking and awarding
            self.swarm = sorted(self.swarm, key=lambda x: -x.fitness)
            best_subswarm = self.swarm[0]
            best_subswarms = self.swarm[:cfg.num_best_ants]                

            # evaporation & update pheromone
            self.network.pheromone = self.network.pheromone*(1-cfg.pheromone_evaporation_coef*min(iter_loop / cfg.warmup_epochs, 1))
            for rank, subswarm in enumerate(best_subswarms):
                if subswarm.finished is False: continue

                if cfg.use_weight_pheromone:
                    weight = subswarm.fitness / global_best_subswarm.fitness if global_best_subswarm else 1
                    weight *= min(iter_loop / cfg.warmup_epochs, 1)
                else: weight = 1
                reward = cfg.Q / (rank + 1) * weight
                
                for ant in subswarm.ants:
                    path = ant.path                  
                    for i in range(len(path) - 1):
                        if cfg.use_sfc_pheromone:
                            self.network.pheromone[ant.sfc.id][path[i]][path[i+1]] += reward
                        else:
                            self.network.pheromone[path[i]][path[i+1]] += reward

            if global_best_subswarm is None:
                global_best_subswarm = best_subswarm
            elif global_best_subswarm.fitness < best_subswarm.fitness:
                global_best_subswarm = best_subswarm
                arg_global_best = iter_loop
            else:
                if iter_loop - arg_global_best >= cfg.early_stopping_epoch:
                    if cfg.display: print("Early stopping at epoch: ", iter_loop)
                    break

            # logger
            if iter_loop % 10 == 0 and cfg.display:
                _num_server = best_subswarm.num_actived_servers
                INFO = f"Iteration {iter_loop}, number of active server: {_num_server}, R_cap: {best_subswarm.R_cap}, R_server: {best_subswarm.R_server}, R_vnf: {best_subswarm.R_vnf}, best score: {best_subswarm.objective_value}"
                print(INFO)
                # for j in self.network.server_ids:
                #     print(f"Server {j}: {self.network.pheromone[0][j][j]}", end=' ')
                # print()

        # update network
        self.network = global_best_subswarm.network
        self.sfc_set = global_best_subswarm.sfc_set
        self.objective_value = global_best_subswarm.objective_value
        self.R_cap = global_best_subswarm.R_cap
        self.R_vnf = global_best_subswarm.R_vnf
        self.R_server = global_best_subswarm.R_server
        self.num_actived_servers = global_best_subswarm.num_actived_servers

        # log
        if cfg.log_path:
            info = "="*10 + "\n" + f"Best at loop {arg_global_best}: SFCs path: {self.sfc_set} - Number of servers: {global_best_subswarm.num_actived_servers} \n Objective value: {global_best_subswarm.objective_value}({global_best_subswarm.R_cap},{global_best_subswarm.R_server},{global_best_subswarm.R_vnf}) \n Node: {self.network.N} \n Link: {self.network.L} \n"
            with open(f"{cfg.log_path}/status.txt", "a") as f:
                f.write(info)
        if (cfg.display): 
            print(f"Number of active server: {global_best_subswarm.num_actived_servers}, best score: {global_best_subswarm.objective_value}({global_best_subswarm.R_cap},{global_best_subswarm.R_server},{global_best_subswarm.R_vnf})")
            print("="*20)

        self.total_time = time.time() - begin
        self.num_success_sfcs = np.sum([1 if sfc.finished else 0 for sfc in self.sfc_set.sfc_set])
        if(cfg.display):
            print("Total time:", self.total_time) 

        if (cfg.log_path):
            info = f"Objective value: {self.objective_value}({self.R_cap},{self.R_server},{self.R_vnf}) | Total time: {self.total_time}"
            with open(f"{cfg.log_path}/status.txt", "a") as f:
                f.write(info)   

    def save_result(self, path, version_name):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, version_name), 'a') as f:
            f.write(f"{self.network.name},{self.sfc_set.name},{self.R_cap},{self.R_server},{self.R_vnf},{self.objective_value},{self.num_success_sfcs},{self.num_actived_servers},{self.total_time} \n")
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)