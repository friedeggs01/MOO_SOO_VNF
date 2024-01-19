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


class NSGA2_HillLambda(NSGA2):

    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)

        self.survival = RankAndHillClimbLambda()

class RankAndHillClimbLambda(Survival):

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
        
        # Replace the first front with better neighborhoods
        hill_lambda(fronts[0], problem, pop)
        
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
    
def hill_lambda(first_front, problem, pop):
    weights = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]

    for ind in first_front:
        min_dist = 10
        best_w = None
        for w in weights:
            distance = abs(w[1]*pop[ind].get("X")[0] - w[0]*pop[ind].get("X")[1]) / np.sqrt(pow(w[0], 2) + pow(w[1], 2))
            if distance < min_dist:
                min_dist = distance
                best_w = w
        while(True):
            # print("pop[ind].X",pop[ind].get("X"))
            print("best_w", best_w)
            print("pop[ind].F", pop[ind].get("F"))      
            f = best_w[0]*pop[ind].get("F")[0] + best_w[1]*pop[ind].get("F")[1]
            
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
            fnei1 = best_w[0]*pop_nei[0].get("F")[0] + best_w[1]*pop_nei[0].get("F")[1] 
            fnei2 = best_w[0]*pop_nei[1].get("F")[0] + best_w[1]*pop_nei[1].get("F")[1]
            print("f", f)
            if fnei1 < f:
                print("fnei1", fnei1)
                print("neighbors_1", neighbors_1.get("X"))
                print("neighbors_1.F", neighbors_1.get("F"))
                pop[ind] = pop_nei[0]
                print("nei1", neighbors_1.get("X"))
                print("pop[ind]", pop[ind].get("X"))
                print("pop[ind].F", pop[ind].get("F"))  
            if fnei2 < f:
                print("fnei2", fnei2)
                print("neighbors_2", neighbors_2.get("X"))
                pop[ind] =  pop_nei[1]
            else: break



parse_doc_string(NSGA2_HillLambda.__init__)
