try:
    import numba
    from numba import jit
except:
    raise Exception("Please install numba to use AGEMOEA: pip install numba")

import numpy as np
import copy

from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.util.misc import has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population



# =========================================================================================================
# Implementation
# =========================================================================================================

class AGE_Hill(AGEMOEA):

    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)

        self.survival = AGEMOEAandHillSurvival()


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------

class AGEMOEAandHillSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective values
        F = pop.get("F")

        N = n_survive

        # Non-dominated sorting
        fronts = self.nds.do(F, n_stop_if_ranked=N)
        
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

        # get max int value
        max_val = np.iinfo(int).max

        # initialize population ranks with max int value
        front_no = np.full(F.shape[0], max_val, dtype=int)

        # assign the rank to each individual
        for i, fr in enumerate(fronts):
            front_no[fr] = i

        pop.set("rank", front_no)

        # get the index of the front to be sorted and cut
        max_f_no = np.max(front_no[front_no != max_val])

        # keep fronts that have lower rank than the front to cut
        selected: np.ndarray = front_no < max_f_no

        n_ind, _ = F.shape

        # crowding distance is positive and has to be maximized
        crowd_dist = np.zeros(n_ind)

        # get the first front for normalization
        front1 = F[front_no == 0, :]

        # follows from the definition of the ideal point but with current non dominated solutions
        ideal_point = np.min(front1, axis=0)

        # Calculate the crowding distance of the first front as well as p and the normalization constants
        crowd_dist[front_no == 0], p, normalization = self.survival_score(front1, ideal_point)
        for i in range(1, max_f_no):  # skip first front since it is normalized by survival_score
            front = F[front_no == i, :]
            m, _ = front.shape
            front = front / normalization
            crowd_dist[front_no == i] = 1. / self.minkowski_distances(front, ideal_point[None, :], p=p).squeeze()

        # Select the solutions in the last front based on their crowding distances
        last = np.arange(selected.shape[0])[front_no == max_f_no]
        rank = np.argsort(crowd_dist[last])[::-1]
        selected[last[rank[: N - np.sum(selected)]]] = True

        pop.set("crowding", crowd_dist)

        # return selected solutions, number of selected should be equal to population size
        return pop[selected]

    def survival_score(self, front, ideal_point):
        front = np.round(front, 12, out=front)
        m, n = front.shape
        crowd_dist = np.zeros(m)

        if m < n:
            p = 1
            normalization = np.max(front, axis=0)
            return crowd_dist, p, normalization

        # shift the ideal point to the origin
        front = front - ideal_point

        # Detect the extreme points and normalize the front
        extreme = find_corner_solutions(front)
        front, normalization = normalize(front, extreme)

        # set the distance for the extreme solutions
        crowd_dist[extreme] = np.inf
        selected = np.full(m, False)
        selected[extreme] = True

        p = self.compute_geometry(front, extreme, n)

        nn = np.linalg.norm(front, p, axis=1)
        distances = self.pairwise_distances(front, p) / nn[:, None]

        neighbors = 2
        remaining = np.arange(m)
        remaining = list(remaining[~selected])
        for i in range(m - np.sum(selected)):
            mg = np.meshgrid(np.arange(selected.shape[0])[selected], remaining, copy=False, sparse=False)
            D_mg = distances[tuple(mg)]  # avoid Numpy's future deprecation of array special indexing

            if D_mg.shape[1] > 1:
                # equivalent to mink(distances(remaining, selected),neighbors,2); in Matlab
                maxim = np.argpartition(D_mg, neighbors - 1, axis=1)[:, :neighbors]
                tmp = np.sum(np.take_along_axis(D_mg, maxim, axis=1), axis=1)
                index: int = np.argmax(tmp)
                d = tmp[index]
            else:
                index = D_mg[:, 0].argmax()
                d = D_mg[index, 0]

            best = remaining.pop(index)
            selected[best] = True
            crowd_dist[best] = d

        return crowd_dist, p, normalization

    @staticmethod
    def compute_geometry(front, extreme, n):
        # approximate p(norm)
        d = point_2_line_distance(front, np.zeros(n), np.ones(n))
        d[extreme] = np.inf
        index = np.argmin(d)

        p = np.log(n) / np.log(1.0 / np.mean(front[index, :]))

        if np.isnan(p) or p <= 0.1:
            p = 1.0
        elif p > 20:
            p = 20.0  # avoid numpy underflow

        return p

    @staticmethod
    @jit(fastmath=True)
    def pairwise_distances(front, p):
        m = np.shape(front)[0]
        distances = np.zeros((m, m))
        for i in range(m):
            distances[i] = np.sum(np.abs(front[i] - front) ** p, 1) ** (1 / p)

        return distances

    @staticmethod
    @jit(fastmath=True)
    def minkowski_distances(A, B, p):
        m1 = np.shape(A)[0]
        m2 = np.shape(B)[0]
        distances = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                distances[i][j] = sum(np.abs(A[i] - B[j]) ** p) ** (1 / p)

        return distances


@jit(nopython=True, fastmath=True)
def find_corner_solutions(front):
    """Return the indexes of the extreme points"""

    m, n = front.shape

    if m <= n:
        return np.arange(m)

    # let's define the axes of the n-dimensional spaces
    W = 1e-6 + np.eye(n)
    r = W.shape[0]
    indexes = np.zeros(n, dtype=numba.intp)
    selected = np.zeros(m, dtype=numba.boolean)
    for i in range(r):
        dists = point_2_line_distance(front, np.zeros(n), W[i, :])
        dists[selected] = np.inf  # prevent already selected to be reselected
        index = np.argmin(dists)
        indexes[i] = index
        selected[index] = True
    return indexes


@jit(fastmath=True)
def point_2_line_distance(P, A, B):
    d = np.zeros(P.shape[0])

    for i in range(P.shape[0]):
        pa = P[i] - A
        ba = B - A
        t = np.dot(pa, ba) / np.dot(ba, ba)
        d[i] = sum((pa - t * ba) ** 2)

    return d


# =========================================================================================================
# Normalization
# =========================================================================================================

def normalize(front, extreme):
    m, n = front.shape

    if len(extreme) != len(np.unique(extreme, axis=0)):
        normalization = np.max(front, axis=0)
        front = front / normalization
        return front, normalization

    # Calculate the intercepts of the hyperplane constructed by the extreme
    # points and the axes

    try:
        hyperplane = np.linalg.solve(front[extreme], np.ones(n))
        if any(np.isnan(hyperplane)) or any(np.isinf(hyperplane)) or any(hyperplane < 0):
            normalization = np.max(front, axis=0)
        else:
            normalization = 1. / hyperplane
            if any(np.isnan(normalization)) or any(np.isinf(normalization)):
                normalization = np.max(front, axis=0)
    except np.linalg.LinAlgError:
        normalization = np.max(front, axis=0)

    normalization[normalization == 0.0] = 1.0

    # Normalization
    front = front / normalization

    return front, normalization

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

parse_doc_string(AGEMOEA.__init__)
