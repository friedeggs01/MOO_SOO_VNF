from ..graph import Network, Link, Node, SFC_SET
from .ant import AntV0
from utils import next_path

import numpy as np
import random
import copy
import os
from tqdm import tqdm
import time

class ACO_VNF_V0():
    def __init__(self, network: Network, sfc_set: SFC_SET):
        self.network = copy.deepcopy(network)
        self.sfc_set = sfc_set
        self.solution = []
        self.ant_swarm = None
        self.sfc_status = [-1 for i in range(len(self.sfc_set.sfc_set))]
        self.total_time = None
        self.objective_value = 3
        self.num_actived_servers = None

    # for all sfc
    def run(self, cfg):
        begin = time.time()

        if cfg.log_path:
            cfg.log_path = next_path(cfg.log_path+"%s")
            os.makedirs(cfg.log_path, exist_ok=True)

        # save config
        # if cfg.log_path:
        #     opt = dict()
        #     for k,v in cfg.__dict__.copy().items():
        #         if "__" not in k: opt[k] = v
        #     json.dump(opt, open(f"{cfg.log_path}/opt.json", "w"), indent=2)

        set_seed(cfg.seed)
        for idx_sfc, sfc in enumerate(self.sfc_set.sfc_set):
            self.network.build_pheromone()
            if cfg.debug: print(f"Service SFC {sfc} ...")
            # aco algorithm
            global_best_ant = None
            arg_global_best = 0

            pbar = tqdm(range(cfg.num_epochs), desc=f"{str(sfc)}") if cfg.display else range(cfg.num_epochs)
            for iter_ant in pbar:
                # initialize params
                self.ant_swarm = [AntV0(id, sfc=sfc, network=self.network, num_sfc=len(self.sfc_set)) for id in range(cfg.num_ant)]
                # each ant do the work
                for ant in self.ant_swarm:
                    ant.run(cfg)

                # ranking and award ants
                fitness_list = np.array([ant.fitness for ant in self.ant_swarm])
                rank = np.argsort(fitness_list)[::-1]
                best_ant = self.ant_swarm[rank[0]]
                best_ants = [self.ant_swarm[idx] for idx in rank[: cfg.num_best_ants]]
                if global_best_ant is None:
                    global_best_ant = best_ant
                elif global_best_ant.fitness < best_ant.fitness:
                    global_best_ant = best_ant
                    arg_global_best = iter_ant
                    
                # evaporation & update pheromone
                self.network.pheromone = self.network.pheromone*(1-cfg.pheromone_evaporation_coef)
                for rank, ant in enumerate(best_ants):
                    if ant.finished is False: continue

                    path = ant.path
                    reward = cfg.Q / (rank + 1)
                    for i in range(len(path) - 1):
                        self.network.pheromone[path[i]][path[i+1]] += reward
                    
                # if best_ant.finished:
                #     path = best_ant.path
                #     reward = best_ant.fitness
                #     for i in range(len(path) - 1):
                #         self.network.pheromone[path[i]][path[i+1]] += cfg.weight_best_ant*reward
                
                if iter_ant - arg_global_best >= cfg.early_stopping_epoch:
                    if cfg.display: print("Early stopping at epoch: ", iter_ant)
                    break     

                # logger
                if iter_ant % 10 == 0 and cfg.display:
                    _num_server = best_ant.num_actived_servers
                    INFO = f"Iteration {iter_ant}, number of active server: {_num_server}, R_cap: {best_ant.R_cap}, R_server: {best_ant.R_server}, R_vnf: {best_ant.R_vnf}, best fitness: {best_ant.fitness}"
                    print(INFO)

            # update network
            self.sfc_status[idx_sfc] = global_best_ant.num_actived_servers
            self.solution.append(global_best_ant.path)
            if global_best_ant.num_actived_servers != -1:
                self.network = global_best_ant.network
                self.objective_value = global_best_ant.objective_value
                self.R_cap = global_best_ant.R_cap
                self.R_vnf = global_best_ant.R_vnf
                self.R_server = global_best_ant.R_server
                
            if cfg.log_path:
                info = "="*20 + "\n" + f"SFC {str(sfc)}, best at iter: {arg_global_best} - vnf location: {global_best_ant.vnf_location} path: {global_best_ant.path} \n Number of servers: {global_best_ant.num_actived_servers}, Objective value: {global_best_ant.objective_value}({global_best_ant.R_cap},{global_best_ant.R_server},{global_best_ant.R_vnf}) \n Server: {self.network.N_server} \n Node: {self.network.N} \n Link: {self.network.L} \n"
                with open(f"{cfg.log_path}/status.txt", "a") as f:
                    f.write(info)

            if (cfg.display): 
                print(f"After requiring sfc {str(sfc)}, number of active server: {global_best_ant.num_actived_servers}, best fitness: {global_best_ant.fitness}")
                print("="*20)

        self.sfc_status = np.array(self.sfc_status)
        self.num_sucesss_sfc = np.sum(self.sfc_status > 0)
        self.num_actived_servers = np.max(self.sfc_status)
        self.total_time = time.time() - begin

        if (cfg.display): 
            print("="*20 + " FINISH" + "="*20)
            print(f"Objective value: {self.objective_value} | Number of SFCs which is serviced: {self.num_sucesss_sfc}/{len(self.sfc_set)} | Total time: {self.total_time}")
        
        if (cfg.log_path):
            info = f"Objective value: {self.objective_value}({self.R_cap},{self.R_server},{self.R_vnf}) | Number of SFCs which is serviced: {self.num_sucesss_sfc}/{len(self.sfc_set)} | Total time: {self.total_time}"
            with open(f"{cfg.log_path}/status.txt", "a") as f:
                f.write(info)
                
    def save_result(self, path, version_name):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, version_name), 'a') as f:
            f.write(f"{self.network.name},{self.sfc_set.name},{self.R_cap},{self.R_server},{self.R_vnf},{self.objective_value},{self.num_sucesss_sfc},{self.num_actived_servers},{self.total_time} \n")

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)