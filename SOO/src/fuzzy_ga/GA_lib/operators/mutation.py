import numpy as np
import random

class SwapMutation():
    def __init__(self, dim, prob=0.5):
        self.dim = dim
        self.prob = prob

    def __call__(self, p: np.ndarray) -> np.ndarray:
        o = np.copy(p)
        for i in range(self.dim):
            if np.random.rand() <= self.prob:
                if i <= 4:
                    o[i] = np.random.uniform(low=0, high=1)
                else:
                    o[i] = np.random.choice([1,2,3])
        return o

class Mutation1():
    def __init__(self, dim):
        self.dim = dim
    def __call__(self, p: np.ndarray) -> np.ndarray:
        t = np.random.randint(0, self.dim)
        o = np.copy(p)
        max_value = np.max(o)
        if(max_value < self.dim - 1):
            o[t] = max_value + 1
        elif(max_value == o[t]):
            o[t] = max_value - 1
        else:
            o[t] = max_value
        o = standarize(o)
        return o

class Mutation2():
    def __init__(self, dim):
        self.dim = dim
    def __call__(self, p: np.ndarray) -> np.ndarray:
        selected_ids = random.sample(range(self.dim), 1)
        o = np.copy(p)
        for id in selected_ids:
            o[id] = np.random.randint(0, max(1,o[id]))
        o = standarize(o)
        return o

def standarize(genes: np.ndarray):
    idx_sorted_genes = np.argsort(genes)
    sorted_genes = np.empty_like(genes)
    rank = 0
    for idx, value in enumerate(idx_sorted_genes):
        sorted_genes[value] = rank
        if(idx > 0 and genes[idx_sorted_genes[idx-1]] < genes[idx_sorted_genes[idx]]):
            rank += 1
            sorted_genes[value] = rank
    return sorted_genes