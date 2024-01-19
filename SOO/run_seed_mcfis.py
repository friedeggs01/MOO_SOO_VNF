import numpy as np
import random
import pandas as pd
import os, glob
import re
from joblib import Parallel, delayed
from argparse import ArgumentParser
from src.graph import Network, SFC_SET
from src.fuzzy_ga import mcFIS

class CFG():
    seed = 42

    # log parameters
    log_path = False
    debug = False
    display = False
    weight_vector = [9,4,4]

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def run_seed(seed_path, out_dir):
    seed = int(seed_path.replace("\\", "/").split("/")[-1])
    cfg = CFG()
    cfg.seed = seed

    set_seed(seed)
    out_path = f"{out_dir}/{seed}"
    os.makedirs(out_path, exist_ok=True)

    for topo_path in sorted(glob.glob(f"{seed_path}/*")): 
        topo = int(topo_path.replace("\\", "/").split("/")[-1])
        print(f"Seed {seed} | Topo {topo}")
        
        with open(f"{topo_path}/log.txt") as f:
            line = f.readlines()[-1]
        
        pattern = r"Best gene: (\[[^]]*\])"
        genes = re.findall(pattern, line)
        genes = genes[0][1:-1].split(",")
        genes = list(map(float, genes))
        # print(genes)
        A, B = np.array(genes[:5]), np.array(genes[5:])

        for dataset in dataset_folders[20*topo: 20*(topo+1)]:
            for i in range(3):
                sfc_set_path = f"./data_with_delay_v2/{dataset}/{type_requests[i]}.txt"
                network = Network(input_path=f"./data_with_delay_v2/{dataset}/input.txt", undirected=True)
                sfc_set = SFC_SET(input_path=sfc_set_path)

                network.create_constraints(sfc_set=sfc_set)
                # sfc_set.create_global_info(network=network)
                
                mcfis = mcFIS(network, sfc_set, cfg)
                # cfg.log_path = f"./{exp_path}/log/{network.name}/{sfc_set.name}/exp"
                mcfis.run(A, B)

                mcfis.save_result(path=out_path, version_name="result.txt")

def create_args():
    parser = ArgumentParser(description="CONFIG VNF")
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--input_dir', default='./experiments/emcfis/test944', type=str)
    parser.add_argument('--output_dir', default='./experiments/mcfis/test20', type=str)
    parser.add_argument('--weight_vector', default= [9,4,4], type=list)
    parser.add_argument('--start_seed', default=13, type=int)
    parser.add_argument('--end_seed', default=43, type=int)
    args = parser.parse_args()
    return args

type_requests = ['request10', 'request20', 'request30']
dataset_folders = sorted(os.listdir("./data_with_delay_v2/"))

if __name__ == "__main__":
    args = create_args()
    os.makedirs(args.output_dir, exist_ok=True)
    Parallel(n_jobs=args.n_jobs)(delayed(run_seed)(seed, args.output_dir) for seed in sorted(glob.glob(f'{args.input_dir}/*'))[::-1])