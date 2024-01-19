from ..object.EA import *
import numpy as np

class ElitismSelection:
    def __init__(self, ascending=False) -> None:
        self.ascending = ascending

    def __call__(self, population: Population, num_selected):
        fitness = population.fitness
        #print(f'Number of inds: {len(fitness)} - fitness:{fitness}')
        ids = np.argsort(np.array(fitness))
        if self.ascending is False:
            ids = ids[::-1]
        return ids[:num_selected]