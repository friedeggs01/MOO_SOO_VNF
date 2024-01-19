import os, csv
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import *
from pymoo.optimize import minimize
from graph.network import Network
from graph.sfc_set import SFC_SET
from pymoo.config import Config
Config.warnings['not_compiled'] = False

names = [ "cogent", "conus", "nsf"]
areas = ["center", "rural", "uniform", "urban"]
requests = [10, 20, 30]
i_s = [0, 1, 2, 3, 4]

folder_path = 'output_moead'
os.makedirs(folder_path, exist_ok=True)
for name in names:
    for area in areas:
        for request in requests:
            for i in i_s:
                name_folder = name+"_"+area+"_"+str(i)
                network = Network("dataset/" + name_folder + "/input.txt")
                sfc_set = SFC_SET("dataset/" + name_folder + "/request" + str(request) + ".txt")
                rs = []
                problem = get_problem("nfv", network, sfc_set)
                ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)
                algorithm = MOEAD(
                    ref_dirs,
                    n_neighbors=15,
                    prob_neighbor_mating=0.7,
                )
                res = minimize(problem,
                                algorithm,
                                ('n_gen', 100),
                                seed=1,
                                verbose=True)                
                file_path = os.path.join(folder_path, f'{name_folder}_request_{str(request)}_result.csv')
                result = {}
                for i in res.F:
                    result['fitness_1'] = i[0]
                    result['fitness_2'] = i[1]
                    with open(file_path, 'a') as file:
                        fieldnames = ['fitness_1', 'fitness_2']
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        writer.writerow(result)