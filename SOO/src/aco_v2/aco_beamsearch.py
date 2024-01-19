from ..graph import Network, Link, Node, SFC_SET
from .ant import AntV2
from .subswarm_ant import SubSwarmAnt
from utils import next_path

import numpy as np
import random
import copy
import json
import os
from tqdm import tqdm
import time
import multiprocessing

class ACO_BEAMSEARCH():
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
            for k,v in cfg.__dict__.copy().items():
                if "__" not in k: opt[k] = v
            json.dump(opt, open(f"{cfg.log_path}/opt.json", "w"), indent=2)

        set_seed(cfg.seed)
        self.num_cpu = min(cfg.num_cpu, multiprocessing.cpu_count()) if cfg.num_cpu else multiprocessing.cpu_count()

        # aco algorithm
        step = 0
        prev_best_figures = [SubSwarmAnt(0, self.network, self.sfc_set, cfg)]
        prev_best_figures[0].fitness = 1 # good for nothing
        prev_best_figures[0].objective_value = 1

        # round loop
        while(True):
            step += 1
            # init pheromone
            for figure in prev_best_figures:
                if cfg.use_sfc_pheromone: 
                    figure.network.build_pheromone(len(self.sfc_set))
                else:
                    figure.network.build_pheromone()

            global_best_subswarms = list()
            arg_global_best = 0

            # aco loop
            prob = np.array([prev_best_figure.fitness for prev_best_figure in prev_best_figures])
            # prob_mean = np.mean(prob)
            # prob_std = np.std(prob)
            # if prob_std > 1e-5 and prob_std < 1:
            #     prob = np.maximum(prob_mean + (prob - prob_mean)/prob_std, 0)
            prob = prob / np.sum(prob)
            
            pbar = tqdm(range(cfg.num_epochs), desc=f"Step {step}") if cfg.display else range(cfg.num_epochs)
            for iter_loop in pbar:
                self.swarm = []
                for i in range(cfg.num_families):
                    choosed_id = np.random.choice(range(len(prev_best_figures)), p=prob)
                    choosed_figure = prev_best_figures[choosed_id]
                    self.swarm.append(SubSwarmAnt(i, choosed_figure.network, choosed_figure.sfc_set, cfg, prev_id_figure=choosed_id))

                if cfg.multiprocessing:
                    with multiprocessing.Pool(self.num_cpu) as p:
                        self.swarm = p.map(multiprocess_ant, self.swarm)
                else:
                    for subswarm in self.swarm:
                        subswarm.run()

                # ranking and awarding
                self.swarm = sorted(self.swarm, key=lambda x: -x.fitness)
                best_subswarm = self.swarm[0]
                best_subswarms = self.swarm[:cfg.num_best_ants]                

                # evaporation & update pheromone
                for figure in prev_best_figures:
                    figure.network.pheromone = figure.network.pheromone*(1-cfg.pheromone_evaporation_coef*min(iter_loop / cfg.warmup_epochs, 1))
                for rank, subswarm in enumerate(best_subswarms):
                    if subswarm.finished is False: continue

                    if cfg.use_weight_pheromone:
                        weight = subswarm.fitness / global_best_subswarms[0].fitness if len(global_best_subswarms) > 0 else 1
                        weight *= min(iter_loop / cfg.warmup_epochs, 1)
                    else: weight = 1
                    reward = cfg.Q / (rank + 1) * weight

                    for ant in subswarm.ants:
                        path = ant.path            
                        for i in range(len(path) - 1):
                            if cfg.use_sfc_pheromone:
                                prev_best_figures[subswarm.prev_id_figure].network.pheromone[ant.sfc.id][path[i]][path[i+1]] += reward
                            else:
                                prev_best_figures[subswarm.prev_id_figure].network.pheromone[path[i]][path[i+1]] += reward

                # update top best
                if len(global_best_subswarms) > 0:
                    if best_subswarm.fitness > global_best_subswarms[0].fitness:
                        arg_global_best = iter_loop
                global_best_subswarms.extend(self.swarm[:cfg.num_best_figure])
                global_best_subswarms = sorted(global_best_subswarms, key=lambda x: -x.fitness)[:cfg.num_best_figure]

                if iter_loop - arg_global_best >= cfg.early_stopping_epoch:
                    if cfg.display: print("Early stopping at epoch: ", iter_loop)
                    if cfg.log_path and cfg.save_all:
                        info = f"Early stopping at epoch: {iter_loop}\n"
                        with open(f"{cfg.log_path}/status.txt", "a") as f:
                            f.write(info)
                    break          

                # logger
                if iter_loop % 10 == 0 and cfg.display:
                    _num_server = best_subswarm.num_actived_servers
                    INFO = f"Iteration {iter_loop}, number of active server: {_num_server}, R_cap: {best_subswarm.R_cap}, R_server: {best_subswarm.R_server}, R_vnf: {best_subswarm.R_vnf}, best objective value: {best_subswarm.objective_value}"
                    print(INFO)

            # update sfc
            for subswarm in global_best_subswarms:
                for ant in subswarm.ants:
                    ant.update_sfc()

            # update network
            self.network = global_best_subswarms[0].network
            self.sfc_set = global_best_subswarms[0].sfc_set
            self.objective_value = global_best_subswarms[0].objective_value
            self.R_cap = global_best_subswarms[0].R_cap
            self.R_vnf = global_best_subswarms[0].R_vnf
            self.R_server = global_best_subswarms[0].R_server
            self.num_actived_servers = global_best_subswarms[0].num_actived_servers
            prev_best_figures = global_best_subswarms

            # log
            if cfg.log_path and cfg.save_all:
                info = "="*10 + f" Step {step}" + "="*10 + "\n" + f"Best at loop {arg_global_best}: SFCs path: {self.sfc_set} - Number of servers: {global_best_subswarms[0].num_actived_servers} \n Objective value: {global_best_subswarms[0].objective_value}({global_best_subswarms[0].R_cap},{global_best_subswarms[0].R_server},{global_best_subswarms[0].R_vnf}) \n Node: {self.network.N} \n Link: {self.network.L} \n"
                with open(f"{cfg.log_path}/status.txt", "a") as f:
                    f.write(info)
            if (cfg.display): 
                print(f"After step {step}, number of active server: {global_best_subswarms[0].num_actived_servers}, best objective: {global_best_subswarms[0].objective_value}({global_best_subswarms[0].R_cap},{global_best_subswarms[0].R_server},{global_best_subswarms[0].R_vnf})")
                print("="*20)

            # check stopping criteria
            stop_crit = True
            for best_subswarm in global_best_subswarms:
                for sfc in best_subswarm.sfc_set.sfc_set:
                    if sfc.finished == False:
                        stop_crit = False
            if stop_crit: break

            if self.objective_value == 3 and iter_loop >= 20: break

        self.total_time = time.time() - begin
        self.num_success_sfcs = np.sum([1 if sfc.finished else 0 for sfc in self.sfc_set.sfc_set])
        if(cfg.display):
            print("Total time:", self.total_time) 

        if cfg.log_path and cfg.save_all:
            info = f"Objective value: {self.objective_value}({self.R_cap},{self.R_server},{self.R_vnf}) | Total time: {self.total_time}"
            with open(f"{cfg.log_path}/status.txt", "a") as f:
                f.write(info)   

    def save_result(self, path, version_name):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, version_name), 'a') as f:
            # f.write(f"{self.network.name},{self.sfc_set.name},{self.R_cap},{self.R_server},{self.R_vnf},{self.objective_value},{self.num_success_sfcs},{self.num_actived_servers},{self.total_time} \n")
            cost = (self.R_server + self.R_vnf) / 2
            f.write(f"{self.R_cap}, {cost}\n")

def multiprocess_ant(subswarm):
    # subswarm = copy.deepcopy(subswarm)
    subswarm.run()
    return subswarm

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)