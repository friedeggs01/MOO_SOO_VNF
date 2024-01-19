from ..graph import Network, Link, Node, SFC_SET
from .ant import Ant
from utils import next_path

import numpy as np
import random
import copy
import json
import os
from tqdm import tqdm
import time
import multiprocessing
class ACO_VNF_V1():
    def __init__(self, network: Network, sfc_set: SFC_SET):
        self.network = copy.deepcopy(network)
        self.sfc_set = sfc_set
        self.num_sfc = len(self.sfc_set)
        self.keypoint_consume = self.sfc_set.keypoint_consume
        self.solution = []
        self.ant_swarm = None
        self.sfc_status = [-1 for i in range(len(self.sfc_set.sfc_set))]
        self.total_time = None
        self.objective_value = 3
        self.num_actived_servers = None
        self.num_cpus = multiprocessing.cpu_count()

    # for all sfc
    def run(self, cfg):
        self.cfg = cfg
        begin = time.time()

        if cfg.log_path:
            cfg.log_path = next_path(cfg.log_path+"%s")
            os.makedirs(cfg.log_path, exist_ok=True)

        # save config
        if cfg.log_path:
            opt = dict()
            for k,v in cfg.__dict__.copy().items():
                if "__" not in k: opt[k] = v
            json.dump(opt, open(f"{cfg.log_path}/opt.json", "w"), indent=2)

        set_seed(cfg.seed)
        
        if cfg.multiprocessing and cfg.display:
            print("Use multiprocessing | Number of cpus: ", self.num_cpus)
            
        for idx_sfc, sfc in enumerate(self.sfc_set.sfc_set):
            self.network.build_pheromone()

            if cfg.debug: print(f"Service SFC {sfc} ...")
            # aco algorithm
            global_best_ant = None
            arg_global_best = 0

            pbar = tqdm(range(cfg.num_epochs), desc=f"{str(sfc)}") if cfg.display else range(cfg.num_epochs)
            for iter_loop in pbar:
                # initialize params
                self.ant_swarm = [Ant(id, sfc=sfc, network=self.network, cfg=cfg, num_sfc=self.num_sfc) for id in range(cfg.num_ants)]

                if cfg.multiprocessing:
                    with multiprocessing.Pool(self.num_cpus) as pool:
                        pool.map(multiprocessing_ant, self.ant_swarm)
                else:
                    # each ant do the work
                    for ant in self.ant_swarm:
                        ant.run()

                # ranking and award ants
                self.ant_swarm = sorted(self.ant_swarm, key=lambda x: -x.fitness)
                best_ant = self.ant_swarm[0]
                best_ants = self.ant_swarm[:cfg.num_best_ants]

                # evaporation & update pheromone
                self.network.pheromone = self.network.pheromone*(1-cfg.pheromone_evaporation_coef)
                for rank, ant in enumerate(best_ants):
                    if ant.finished is False: continue
                    path = ant.path
                    if cfg.use_weight_pheromone:
                        weight = ant.fitness / global_best_ant.fitness if global_best_ant is not None else 1
                        weight *= min(iter_loop / cfg.warmup_epochs, 1)
                    else: weight = 1

                    reward = cfg.Q / (rank + 1) * weight
                    for i in range(len(path) - 1):
                        self.network.pheromone[path[i]][path[i+1]] += reward
                
                if global_best_ant is None:
                    global_best_ant = best_ant
                elif global_best_ant.fitness < best_ant.fitness:
                    global_best_ant = best_ant
                    arg_global_best = iter_loop
                else:
                    if iter_loop - arg_global_best >= cfg.early_stopping_epoch:
                        if cfg.display: print("Early stopping at epoch: ", iter_loop)
                        break

                # logger
                if iter_loop % 10 == 0 and cfg.display:
                    _num_server = best_ant.num_actived_servers
                    INFO = f"Iteration {iter_loop}, number of active server: {_num_server}, R_cap: {best_ant.R_cap}, R_server: {best_ant.R_server}, R_vnf: {best_ant.R_vnf}, best objective: {best_ant.objective_value}"
                    print(INFO)
                    print(self.ant_swarm)

            # update network
            self.sfc_status[idx_sfc] = global_best_ant.num_actived_servers
            self.solution.append(global_best_ant.path)
            if global_best_ant.num_actived_servers != -1:
                self.network = global_best_ant.network
                self.objective_value = global_best_ant.objective_value
                self.R_cap = global_best_ant.R_cap
                self.R_vnf = global_best_ant.R_vnf
                self.R_server = global_best_ant.R_server
                
                # if self.network.N[sfc.destination].type == 0:
                #     self.keypoint_consume[sfc.destination] -= sfc.memory

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
      
def multiprocessing_ant(ant):
    print(ant)
    ant.run()

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
