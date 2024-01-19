import numpy as np
import copy
from ..tasks.task import AbstractTask

class Individual:
    def __init__(self, genes: np.ndarray, task: AbstractTask):
        self.genes = genes
        self.dim = len(genes)
        self.fitness, self.objective_list = task(self.genes)

    def __repr__(self) -> str:
        return f"Genes: {self.genes}"

class Population:
    def __init__(self, num_inds: int, task: AbstractTask) -> None:
        self.num_inds = num_inds
        self.task = task
        self.dim = task.dim
        self.population = [Individual(self.task.generate_gene(), self.task) for i in range(num_inds)]
        self.fitness: list
        self.eval()

    def __getRWSIndividual__(self, size: int):
        # update fitness
        if self.num_inds != len(self.fitness):
            self.eval()
        
        p = np.array(self.fitness)
        p = p / np.sum(p)
        ids = np.random.choice(a=np.arange(0,self.num_inds), p=p, size=size, replace=False)
        output = [self.population[id] for id in ids]
        return output

    def __getRandomIndividual__(self, size: int):
        output = []
        ids = np.random.randint(low=0,high=self.num_inds,size=size)
        for i in ids:
            output.append(self.population[i])
        return output

    def __addIndividual__(self, ind: Individual):
        self.population.append(ind)
        self.fitness.append(ind.fitness)
        self.num_inds += 1

    def __getBestIndividiual__(self, max=True):
        idx = np.argmax(self.fitness) if max else np.argmin(self.fitness)
        return self.population[idx]

    def eval(self):
        self.fitness = [ind.fitness for ind in self.population]

    def __len__(self):
        return self.num_inds

    def __getGenes__(self):
        output = []
        for ind in self.population:
            output.append(ind.genes)
        return output



