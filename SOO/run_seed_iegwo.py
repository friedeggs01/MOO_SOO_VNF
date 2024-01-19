import numpy as np
import pandas as pd

import os, glob, shutil
import json
from joblib import Parallel, delayed
from argparse import ArgumentParser

from src.graph import *
from src.ie_gwo import IEGWO
from utils.algorithms import *

class CFG():
    def __init__(self):
        # loop parameters
        self.num_epochs = 60
        self.num_wolfs = 20

        # wolf params
        self.exploit_alpha, self.exploit_beta, self.exploit_delta = 0.2, 0.2, 0.2
        self.exploit_ab = self.exploit_alpha + self.exploit_beta
        self.exploit_abd = self.exploit_ab + self.exploit_delta
        self.exploit_nsm = 1 - self.exploit_abd
        self.explore_epsilon = 0.005
        self.threshold_state_lower, self.threshold_state_upper = 0.3, 1
        
        # log parameters
        self.save_all = False
        self.log_path = None
        self.debug = False
        self.display = False
        
        self.weight_vector = [9, 4, 4]
        
def run_seed(seed, out_dir):
    cfg = CFG()
    cfg.seed = seed
    # out_dir = args.output_dir
    string_list = [str(num) for num in args.weight_vector]
    my_string = ''.join(string_list)
    out_dir = args.output_dir + my_string
    exp_path = f"{out_dir}/{seed}"
    print("Seed: ", seed)
    print("Exp path:", exp_path)
    
    for dataset in dataset_folders:
        for i in range(3):
            sfc_set_path = f"./data_with_delay_v2/{dataset}/{type_requests[i]}.txt"
            for cnt in range(1):
                network = Network(input_path=f"./data_with_delay_v2/{dataset}/input.txt", undirected=True)
                sfc_set = SFC_SET(input_path=sfc_set_path)

                network.create_constraints(sfc_set=sfc_set)
                # sfc_set.create_global_info(network=network)

                iegwo = IEGWO(network=network, sfc_set=sfc_set, cfg=cfg)
                cfg.log_path = f"./{exp_path}/log/{network.name}/{sfc_set.name}/exp"
                iegwo.run()

                iegwo.save_result(path=exp_path, version_name="result.txt")

def create_args():
    parser = ArgumentParser(description="CONFIG VNF")
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--output_dir', default='./experiments/iegwo/test', type=str)
    parser.add_argument('--start_seed', default=13, type=int)
    parser.add_argument('--end_seed', default=43, type=int)
    parser.add_argument('--weight_vector', default= [9, 4, 4], type=list)
    args = parser.parse_args()
    return args

type_requests = ['request10', 'request20', 'request30']
dataset_folders = sorted(os.listdir("./data_with_delay_v2/"))

if __name__ == "__main__":
    args = create_args()
    Parallel(n_jobs=args.n_jobs)(delayed(run_seed)(seed, args.output_dir) for seed in range(args.start_seed,args.end_seed)[::-1])