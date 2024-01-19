import numpy as np
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


class NSGA2_OneHill(NSGA2):

    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)
        self.survival = RankAndOneHillClimb()


class RankAndOneHillClimb(Survival):

    def __init__(self, nds=None, crowding_func="cd"):

        crowding_func_ = get_crowding_function(crowding_func)
        self.local_search = None
        
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
        
        # Individual after hill need to replace
        pop = self.hill_climbing(problem, pop, fronts[0])

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
    
    def hill_climbing(self, problem, pop, first_front):
        print("len first front", len(first_front))
        if self.local_search is None:

            # random a element fromt frist_front, then find this from pop
            # print("len first front", len(first_front))
            index = np.random.randint(0, len(first_front))
            ind = first_front[index]
            self.local_search = pop[ind]
        else:
            # Find index if known individual
            target_individual = self.local_search
            ind = -1
            for i, individual in enumerate(pop):
                if individual == target_individual:
                    ind = i
                    break
        # print(self.local_search.get("X")) 
        # print(self.local_search)
        # neighborhood_1: permutate two random position
        neighbors_1 = copy.deepcopy(pop[ind])
        index1, index2 = np.random.choice(len(neighbors_1.X), 2, replace=False)
        temp = neighbors_1.X[index1]
        neighbors_1.X[index1] = neighbors_1.X[index2]
        neighbors_1.X[index2] = temp

        # neighborhood_2: permutate two random server
        neighbors_2 = copy.deepcopy(pop[ind])
        index1, index2 = np.random.choice(problem.num_server, 2, replace=False)
        for i in range(problem.vnf_max):
            temp = neighbors_2.X[index1*problem.vnf_max+i]
            neighbors_2.X[index1*problem.vnf_max+i] = neighbors_2.X[index2*problem.vnf_max+i]
            neighbors_2.X[index2*problem.vnf_max+i] = temp  

        X = np.vstack((neighbors_1.get("X"), neighbors_2.get("X")))
        # print(neighbors_1.get("X"))
        # print(neighbors_2.get("X"))
        # print(X)
        pop_nei = Population.new("X", X)
        Evaluator().eval(problem, pop_nei)
        
        # print("individual ", pop[ind].get("X"))
        # print("neighbor1", pop_nei[0].get("X"))
        # print("neighbor1", pop_nei[1].get("X"))
        if pop_nei[0].get("F")[0] > pop[ind].get("F")[0] and pop_nei[0].get("F")[1] > pop[ind].get("F")[1]:
            pop[ind] = pop_nei[0]
            self.local_search = pop_nei[0]
        elif pop_nei[1].get("F")[0] > pop[ind].get("F")[0] and pop_nei[1].get("F")[1] > pop[ind].get("F")[1]:
            pop[ind] = pop_nei[1]
            self.local_search = pop_nei[1]
        else:
            if np.random.random() < 0.5:
                pop[ind] = pop_nei[0]
                self.local_search = pop_nei[0]
            else:
                pop[ind] = pop_nei[1]
                self.local_search = pop_nei[1]
        # print(self.local_search.get("X"))       
        return pop


parse_doc_string(NSGA2_OneHill.__init__)
