import numpy as np
import copy
import random

class UniformCrossover():
    def __init__(self, dim, prob=0.5):
        self.dim = dim
        self.prob = prob
    def __call__(self, p1: np.ndarray, p2: np.ndarray):
        o1 = []
        o2 = []

        for i in range(self.dim):
            if np.random.rand() <= self.prob:
                o1.append(p2[i])
                o2.append(p1[i])
            else:
                o1.append(p1[i])
                o2.append(p2[i])
        o1 = np.array(o1)
        o2 = np.array(o2)
        return o1, o2
        

# class OnePointCrossover():
#     def __init__(self, dim: int):
#         self.dim = dim
#     def __call__(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         t1 = np.random.randint(0, self.dim)
#         o1 = np.empty_like(p1)
#         o2 = np.empty_like(p2)
#         o1[:t1], o1[t1:] = p1[:t1], p2[t1:]
#         o2[:t1], o2[t1:] = p2[:t1], p1[t1:]
#         o1 = standarize(o1)
#         o2 = standarize(o2)
#         return o1, o2

# class TwoPointCrossover():
#     def __init__(self, dim: int):
#         self.dim = dim
#     def __call__(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         t1 = np.random.randint(0, self.dim - 1)
#         t2 = np.random.randint(t1 + 1, self.dim)
#         o1 = np.empty_like(p1)
#         o2 = np.empty_like(p2)
#         o1[:t1], o1[t1:t2], o1[t2:] = p1[:t1], p2[t1:t2], p1[t2:]
#         o2[:t1], o2[t1:t2], o2[t2:] = p2[:t1], p1[t1:t2], p2[t2:]
#         o1 = standarize(o1)
#         o2 = standarize(o2)
#         return o1, o2

# def standarize(genes: np.ndarray):
#     idx_sorted_genes = np.argsort(genes)
#     sorted_genes = np.empty_like(genes)
#     rank = 0
#     for idx, value in enumerate(idx_sorted_genes):
#         sorted_genes[value] = rank
#         if(idx > 0 and genes[idx_sorted_genes[idx-1]] < genes[idx_sorted_genes[idx]]):
#             rank += 1
#             sorted_genes[value] = rank
#     return sorted_genes
