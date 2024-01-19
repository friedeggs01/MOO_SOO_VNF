import numpy as np

from pymoo.core.crossover import Crossover

class VNFcx(Crossover):
    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, problem, X, **kwargs):
        num_server = problem.num_server
        vnf_max = problem.vnf_max
        num_vnf = problem.num_vnf
        _, n_matings, n_var = X.shape
        Xp = np.empty((2, n_matings, num_server * vnf_max), dtype=X.dtype)

        for j in range(n_matings):
            p1 = X[0][j]
            p2 = X[1][(j + 1) % n_matings]  

            c1 = np.full_like(p1, -1)
            
            for vnf in range(num_vnf):
                positions = np.where(p1 == vnf)[0]
                if positions.size > 0:
                    c1[positions[0]] = vnf
            
            for position in range(len(c1)):
                if c1[position] == -1:
                    c1[position] = p2[position]
            for ser in range(num_server):
                start = ser * vnf_max
                end = start + vnf_max - 1
                values_in_range = c1[start:end+1]
                unique_values, counts = np.unique(values_in_range, return_counts=True)
                repeated_values = unique_values[counts > 1]
                for value in repeated_values:
                    indices = np.where(values_in_range == value)[0]
                    if len(indices) >= 2:
                        c1[start + indices[1]] = num_vnf

            Xp[0][j, :] = c1
            Xp[1][(j+1)%n_matings, :] = c1
        
        return Xp