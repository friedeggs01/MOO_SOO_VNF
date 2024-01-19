from src.fuzzy_ga.GA_lib.operators.mutation import SwapMutation
from src.fuzzy_ga.GA_lib.operators.crossover import UniformCrossover
from src.fuzzy_ga.GA_lib.operators.selection import ElitismSelection
from src.fuzzy_ga.GA_lib.tasks.mcFIS_task import mcFISTask 
from src.fuzzy_ga.GA_lib.models.GA_mcFIS import GA_mcFIS
from src.fuzzy_ga.GA_lib.object.EA import Population, Individual
from src.graph import Network, SFC_SET

import numpy as np
from argparse import ArgumentParser
import random
import os
from joblib import Parallel, delayed

class CFG():
    seed = 42

    # log parameters
    log_path = None
    debug = False
    display = False
    weight_vector = [5,5,5]

def run_seed(seed, out_dir):
    cfg = CFG()
    cfg.seed = seed
    # out_dir = args.output_dir
    string_list = [str(num) for num in args.weight_vector]
    my_string = ''.join(string_list)
    out_dir = args.output_dir + my_string
    out_path = f"{out_dir}/{seed}"
    os.makedirs(f"{out_path}", exist_ok=True)

    print("Seed:", seed)
    network_lst = []
    sfc_set_lst = []

    for topology in range(3):
        out_path = f"{out_dir}/{seed}/{topology}"
        os.makedirs(f"{out_path}", exist_ok=True)

        print("Out path:", out_path)
        for id_network in range(20*topology, 20*(topology+1)):
            if id_network % 5 == 0:
                for request in type_requests:
                    dataset = dataset_folders[id_network]
                    sfc_set_path = f"./data_with_delay_v2/{dataset}/{request}.txt"
                    network = Network(input_path=f"./data_with_delay_v2/{dataset}/input.txt", undirected=True)
                    sfc_set = SFC_SET(input_path=sfc_set_path)

                    network.create_constraints(sfc_set=sfc_set)
                    
                    network_lst.append(network)
                    sfc_set_lst.append(sfc_set)

        nb_inds = 30
        nb_generations = 100
        mcfistask = mcFISTask(dimA=5, dimB=5, networks=network_lst, sfc_sets=sfc_set_lst, cfg=cfg)
        crossover = UniformCrossover(dim=mcfistask.dim, prob=0.3)
        mutation = SwapMutation(dim=mcfistask.dim, prob=0.3)
        selection = ElitismSelection(ascending=False)

        model = GA_mcFIS(seed=cfg.seed)
        model.compile(mcfistask, crossover, mutation, selection)
        model.fit(nb_generations=nb_generations, nb_inds=nb_inds, p_c=0.5, p_m=0.4, p_r=0.1, out_path=
                  out_path)
        
def create_args():
    parser = ArgumentParser(description="CONFIG VNF")
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--output_dir', default='./experiments/emcfis/test', type=str)
    parser.add_argument('--start_seed', default=13, type=int)
    parser.add_argument('--end_seed', default=43, type=int)
    parser.add_argument('--weight_vector', default= [5,5,5], type=list)
    args = parser.parse_args()
    return args

type_requests = ['request10', 'request20', 'request30']
dataset_folders = sorted(os.listdir("./data_with_delay_v2/"))

if __name__ == "__main__":
    args = create_args()
    Parallel(n_jobs=args.n_jobs)(delayed(run_seed)(seed, args.output_dir) for seed in range(args.start_seed,args.end_seed)[::-1])
