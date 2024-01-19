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


class KNSGA2_Hill(NSGA2):

    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)

        self.survival = RankAndKnee_HillClimb()

class RankAndKnee_HillClimb(Survival):

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
        print(len(F))
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
        print(len(F))
        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        # find knee points for each front
        survivors.extend(find_knee_points(fronts, problem, F))
        
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
        pop_nei.append(neighbors_1.X)

    pop_nei = np.array(pop_nei)
    return pop_nei

def find_knee_points(fronts, problem, F):

    knee_points = []
    for front in fronts:
        # Identify the extreme points for each objective.
        extreme_points = []
        
        for objective_index in range(problem.n_obj):
            min_value = min(F[individual][objective_index] for individual in front)
            max_value = max(F[individual][objective_index] for individual in front)
            extreme_points.append((min_value, max_value))

        # Connect the extreme points for each objective to form a line.
        pareto_front = []
        for i in range(len(extreme_points) - 1):
            pareto_front.append(np.linspace(extreme_points[i][0], extreme_points[i][1], 100))

        # Calculate the distance of each individual on the front to the Pareto front.
        distances = []
        for individual in front:
            distance = np.min([np.linalg.norm(individual - point) for point in pareto_front])
            distances.append(distance)

        # Find the individual with the highest distance to the Pareto front.
        knee_point_index = np.argmax(distances)
        knee_point = front[knee_point_index]

        knee_points.append(knee_point)

    return knee_points

parse_doc_string(KNSGA2_Hill.__init__)
