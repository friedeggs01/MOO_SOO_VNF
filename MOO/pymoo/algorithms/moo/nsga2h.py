import numpy as np
import random
import copy
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

# =========================================================================================================
# Implementation
# =========================================================================================================


class NSGA2_H(NSGA2):

    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)

        self.survival = RankAndHillClimb()

class RankAndHillClimb(Survival):

    def __init__(self, nds=None, crowding_func="cd"):

        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_

    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):
        
        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        
        # Individual(s) after hill need to append
        additional_indi = hill_climbing(fronts[0], problem, pop)

        # Append individuals to the existing population
        X = np.concatenate((pop.get("X"), additional_indi))
        pop = Population.new("X", X)
        Evaluator().eval(problem, pop)
        
        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=n_remove
                    )

                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]
    
def hill_climbing(first_front, problem, pop):
    pop_nei = []
    for ind in first_front:

        # neighborhood_1: permutate two random position
        neighbors_1 = copy.deepcopy(pop[ind])
        index1, index2 = np.random.choice(len(neighbors_1.X), 2, replace=False)
        temp = neighbors_1.X[index1]
        neighbors_1.X[index1] = neighbors_1.X[index2]
        neighbors_1.X[index2] = temp
        pop_nei.append(neighbors_1.X)

        # neighborhood_2: permutate two random server
        neighbors_2 = copy.deepcopy(pop[ind])
        index1, index2 = np.random.choice(problem.num_server, 2, replace=False)
        for i in range(problem.vnf_max):
            temp = neighbors_2.X[index1*problem.vnf_max+i]
            neighbors_2.X[index1*problem.vnf_max+i] = neighbors_2.X[index2*problem.vnf_max+i]
            neighbors_2.X[index2*problem.vnf_max+i] = temp            
        pop_nei.append(neighbors_2.X)
        
        # neigborhood_3: permutate a group vnf of two random server
        neighbors_3 = copy.deepcopy(pop[ind])
        index1, index2 = np.random.choice(problem.num_server, 2, replace=False)
        # print("vnf_max", problem.vnf_max)
        i_vnf1, i_vnf2 = np.random.choice(problem.vnf_max, 2, replace=True)
        if i_vnf2 <= i_vnf1:
            i_vnf2 = i_vnf1 + 1
        for i in range(i_vnf1, i_vnf2):
            temp = neighbors_3.X[index1 * problem.vnf_max + i]
            neighbors_3.X[index1 * problem.vnf_max + i] = neighbors_3.X[index2 * problem.vnf_max + i]
            neighbors_3.X[index2 * problem.vnf_max + i] = temp              
        pop_nei.append(neighbors_3.X)
        
        # neigborhood_4: find a position empty then randomly move a vnf to this
        neighbors_4 = copy.deepcopy(pop[ind])
        positions = [i for i, x in enumerate(neighbors_4.X) if x == problem.vnf_max]
        if len(positions) > 0:
            neighbors_4.X[positions] = random.randint(0, problem.vnf_max)             
        pop_nei.append(neighbors_4.X)

    pop_nei = np.array(pop_nei)
    return pop_nei


parse_doc_string(NSGA2_H.__init__)
