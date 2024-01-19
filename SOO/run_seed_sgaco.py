import numpy as np
import pandas as pd
import time

import os, glob, shutil
import json
from joblib import Parallel, delayed
from argparse import ArgumentParser

from src.graph import *
from src.aco_v2 import ACO_VNF_V2, ACO_BEAMSEARCH
from src.ie_gwo import IEGWO
from utils.algorithms import *

class CFG():
    def __init__(self):
        # loop parameters
        self.num_epochs = 60
        self.warmup_epochs = 20
        self.early_stopping_epoch = 40
        self.multiprocessing = False
        self.num_cpu = 5
        
        # ant-related parameters
        self.num_best_figure = 10
        self.num_families = 40
        self.num_best_ants = 10
        self.weight_best_ant = 0

        self.pheromone_evaporation_coef = 0.05
        self.use_sfc_pheromone = True
        self.use_weight_pheromone = True

        # hyperparameters
        self.num_dijkstra_paths = 1
        self.seed = 42
        self.Q = 0.1
        self.threshold_state_lower, self.threshold_state_upper = 0.3, 1
        self.alpha, self.beta = 6, 4
        self.if_weight_subpath_list = [[2,1,1,1],[2,1,1,1],[2,1,1,1]]
        self.if_weight_vnf_list = [[2,1,1],[2,1,1],[2,1,1]]
        self.weight_vector = [3,8,8]

        
        # log parameters
        self.log_path = False
        self.debug = False
        self.display = False
        self.save_all = False

def run_seed(seed, args):
    # out_dir = args.output_dir
    string_list = [str(num) for num in args.weight_vector]
    my_string = ''.join(string_list)
    out_dir = args.output_dir + my_string
    # print("Seed: ", seed)
    exp_path = f"{out_dir}/{seed}"
    print("Exp path:", exp_path)
    
    # save config
    cfg = CFG()
    cfg.seed = seed
    cfg.num_best_figure = args.beam_width
    cfg.num_families = args.num_subswarms
    cfg.num_epochs = args.num_epochs
    cfg.use_weight_pheromone = args.use_weight_pheromone
    cfg.use_sfc_pheromone = args.use_sfc_pheromone
    cfg.weight_vector = args.weight_vector

    os.makedirs(exp_path, exist_ok=True)
    opt = dict()
    for k,v in cfg.__dict__.copy().items():
        if "__" not in k: opt[k] = v
    json.dump(opt, open(f"{exp_path}/opt.json", "w"), indent=2)
    
    for dataset in dataset_folders:
        id_dataset = int(dataset[-2:])
        if id_dataset < 20:
            cfg.if_weight_subpath = cfg.if_weight_subpath_list[0]
            cfg.if_weight_vnf = cfg.if_weight_vnf_list[0]
        elif 20 <= id_dataset < 40:
            cfg.if_weight_subpath = cfg.if_weight_subpath_list[1]
            cfg.if_weight_vnf = cfg.if_weight_vnf_list[1]
        elif 40 <= id_dataset < 60:
            cfg.if_weight_subpath = cfg.if_weight_subpath_list[2]
            cfg.if_weight_vnf = cfg.if_weight_vnf_list[2]

        for i in range(3):
            sfc_set_path = f"./data_with_delay_v2/{dataset}/{type_requests[i]}.txt"
            network = Network(input_path=f"./data_with_delay_v2/{dataset}/input.txt", undirected=True)
            sfc_set = SFC_SET(input_path=sfc_set_path)

            network.create_constraints(sfc_set=sfc_set)
            sfc_set.create_global_info(network=network)

            aco_vnf = ACO_BEAMSEARCH(network=network, sfc_set=sfc_set)
            cfg.log_path = f"./{exp_path}/log/{network.name}/{sfc_set.name}/exp"
            aco_vnf.run(cfg)

            aco_vnf.save_result(path=exp_path, version_name="result.txt")

def create_args():
    parser = ArgumentParser(description="CONFIG VNF")
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--output_dir', default='./experiments/aco_v2/test', type=str)
    parser.add_argument('--start_seed', default=13, type=int)
    parser.add_argument('--end_seed', default=43, type=int)
    parser.add_argument('--weight_vector', default= [3,8,8], type=list)
    
    # ACO hyperparameters
    parser.add_argument('--beam_width', default=10, type=int)
    parser.add_argument('--num_subswarms', default=40, type=int)
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--use_sfc_pheromone', default=1, type=int)
    parser.add_argument('--use_weight_pheromone', default=1, type=int)
    args = parser.parse_args()
    return args

type_requests = ['request10', 'request20', 'request30']
dataset_folders = sorted(os.listdir("./data_with_delay_v2/"))

if __name__ == "__main__":
    args = create_args()
    Parallel(n_jobs=args.n_jobs)(delayed(run_seed)(seed, args) for seed in range(args.start_seed,args.end_seed)[::-1])