import numpy as np
import random
import copy
import os
from tqdm import tqdm
import time
import json

from .wolf import Wolf
from .operators import *
from ..graph import Network, SFC_SET
from utils import next_path

class IEGWO():
    def __init__(self, network: Network, sfc_set: SFC_SET, cfg):
        self.cfg = cfg
        self.network = copy.deepcopy(network)
        self.sfc_set = copy.deepcopy(sfc_set)
        self.sfc_status = np.zeros(len(self.sfc_set.sfc_set), dtype=np.int16)
        self.num_sfc = len(self.sfc_set)

    def run(self):
        begin = time.time()
        if self.cfg.log_path:
            self.cfg.log_path = next_path(self.cfg.log_path+"%s")
            os.makedirs(self.cfg.log_path, exist_ok=True)

        # save config
        if self.cfg.log_path:
            opt = dict()
            for k,v in self.cfg.__dict__.copy().items():
                if "__" not in k: opt[k] = v
            json.dump(opt, open(f"{self.cfg.log_path}/opt.json", "w"), indent=2)

        set_seed(self.cfg.seed)

        for iter_sfc, sfc in enumerate(self.sfc_set.sfc_set):
            if self.cfg.debug: print(f"Service SFC {sfc} ...")
            # initialize wolf swarm
            self.population = [Wolf(id, self.network, sfc, self.cfg, num_sfc=self.num_sfc) for id in range(self.cfg.num_wolfs)]
            for wolf in self.population:
                wolf.cal_fitness()
            self.population = sorted(self.population, key=lambda x: x.fitness)
            alpha_wolf, beta_wolf, delta_wolf = self.population[:3]
            global_alpha_wolf = alpha_wolf
            arg_global_best = 0

            # initialize params
            len_vector = len(sfc.vnf_list)
            explore_servers = self.network.server_ids
            server2idx = dict()
            for idx, _s in enumerate(explore_servers): server2idx[_s] = idx
            explore_servers_prob = np.ones(len(explore_servers)) / len(explore_servers)

            pbar = tqdm(range(self.cfg.num_epochs), desc=str(sfc)) if self.cfg.display else range(self.cfg.num_epochs)
            for iter_loop in pbar:
                if alpha_wolf.feasible:
                    for _s in alpha_wolf.vector:
                        explore_servers_prob[server2idx[_s]] += self.cfg.explore_epsilon
                    explore_servers_prob /= np.sum(explore_servers_prob)
                for iter_wolf, wolf in enumerate(self.population):
                    # adaptive role playing
                    role_prob = alpha_wolf.fitness / wolf.fitness * ((iter_loop + self.cfg.num_epochs)/(self.cfg.num_epochs*2)) 
                    if np.random.rand() >= role_prob: # exploit
                        vector = np.zeros(len_vector, dtype=np.int16)
                        for i in range(len_vector):
                            rnd_prob = np.random.rand()
                            if rnd_prob <= self.cfg.exploit_alpha:
                                vector[i] = alpha_wolf.vector[i]
                            elif rnd_prob <= self.cfg.exploit_ab:
                                vector[i] = beta_wolf.vector[i]
                            elif rnd_prob <= self.cfg.exploit_abd:
                                vector[i] = delta_wolf.vector[i]
                            else:
                                current_vnf = sfc.vnf_list[i]
                                candidate_servers = [
                                    server for server in self.network.N_server 
                                    if server.cpu_available >= sfc.cpu and server.check_num_vnf(current_vnf)
                                ]

                                # criteria 1: whether this vnf is installed or not
                                if1 = np.array([random.uniform(a=self.cfg.threshold_state_upper, b=1) if current_vnf in server.vnf_used 
                                                else self.cfg.threshold_state_lower*self.network.min_cost_vnfs_axis_server[current_vnf]/server.vnf_cost[current_vnf]
                                                for server in candidate_servers])
                                if1 = if1 / np.sum(if1)

                                # criteria 2: about state server
                                if2 = np.array([random.uniform(a=self.cfg.threshold_state_upper, b=1) if (len(server.vnf_used) > 0) 
                                                else self.cfg.threshold_state_lower*self.network.min_cost_servers/server.cost
                                                for server in candidate_servers])
                                if2 = if2 / np.sum(if2)
                            
                                # criteria 3: about delay
                                if3 = 1 / np.array([server.delay for server in candidate_servers])
                                if3 = if3 / np.sum(if3)

                                prob_nsm = (if1 + if2 + if3) / 3
                                selected_server = np.random.choice(candidate_servers, p=prob_nsm)
                                vector[i] = selected_server.id
                    else: # explore
                        vector = np.random.choice(explore_servers, size=len_vector, p=explore_servers_prob)

                    new_wolf = Wolf(wolf.id, self.network, sfc, self.cfg, self.num_sfc ,vector)
                    if new_wolf.cal_fitness() <= wolf.fitness:
                        self.population[iter_wolf] = new_wolf
                        
                self.population = sorted(self.population, key=lambda x: x.fitness)
                alpha_wolf, beta_wolf, delta_wolf = self.population[:3]
                if global_alpha_wolf.fitness > alpha_wolf.fitness:
                    global_alpha_wolf = alpha_wolf
                    arg_global_best = iter_loop
                else:
                    if iter_loop - arg_global_best >= self.cfg.num_epochs // 2:
                        if self.cfg.display: print("Early stopping at epoch: ", iter_loop)
                        break          
                if iter_loop % 10 == 0 and self.cfg.display:
                    print(f"Iter {iter_loop} - Alpha wolf: {alpha_wolf}")
        
            # update network
            if global_alpha_wolf.feasible:
                global_alpha_wolf.update_sfc()
                self.network = global_alpha_wolf.network
                self.fitness = global_alpha_wolf.fitness
                self.R_cap = global_alpha_wolf.R_cap
                self.R_vnf = global_alpha_wolf.R_vnf
                self.R_server = global_alpha_wolf.R_server
                self.sfc_status[iter_sfc] = self.network.count_actived_servers()

            if self.cfg.log_path and self.cfg.save_all:
                info = "="*20 + "\n" + f"SFC {str(sfc)}, best at iter: {arg_global_best} - vnf location: {global_alpha_wolf.vector} path: {global_alpha_wolf.path} \n Number of servers: {self.sfc_status[iter_sfc]}, Fitness: {global_alpha_wolf.fitness}({global_alpha_wolf.R_cap},{global_alpha_wolf.R_server},{global_alpha_wolf.R_vnf}) \n Server: {self.network.N_server} \n Node: {self.network.N} \n Link: {self.network.L} \n"
                with open(f"{self.cfg.log_path}/status.txt", "a") as f:
                    f.write(info)

            if self.cfg.display:
                print(f"After requiring sfc {str(sfc)}, number of active server: {self.sfc_status[iter_sfc]}, best fitness: {global_alpha_wolf.fitness}")
                print("="*20)

        # update final
        self.sfc_status = np.array(self.sfc_status)
        self.num_sucesss_sfc = np.sum(self.sfc_status > 0)
        self.num_actived_servers = np.max(self.sfc_status)
        self.total_time = time.time() - begin
        self.objective_value = self.fitness

        if (self.cfg.display): 
            print("="*20 + " FINISH" + "="*20)
            print(f"Fitness: {self.fitness} | Number of SFCs which is serviced: {self.num_sucesss_sfc}/{len(self.sfc_set)} | Total time: {self.total_time}")
        
        if self.cfg.log_path and self.cfg.save_all:
            info = f"Fitness: {self.fitness}({self.R_cap},{self.R_server},{self.R_vnf}) | Number of SFCs which is serviced: {self.num_sucesss_sfc}/{len(self.sfc_set)} | Total time: {self.total_time}"
            with open(f"{self.cfg.log_path}/status.txt", "a") as f:
                f.write(info)

    def save_result(self, path, version_name):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, version_name), 'a') as f:
            # f.write(f"{self.network.name},{self.sfc_set.name},{self.R_cap},{self.R_server},{self.R_vnf},{self.objective_value},{self.num_success_sfcs},{self.num_actived_servers},{self.total_time} \n")
            cost = (self.R_server + self.R_vnf) / 2
            f.write(f"{self.R_cap}, {cost}\n")

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)