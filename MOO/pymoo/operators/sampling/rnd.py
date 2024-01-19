import numpy as np

from pymoo.core.sampling import Sampling


def random(problem, n_samples=1):
    X = np.random.random((n_samples, problem.n_var))

    if problem.has_bounds():
        xl, xu = problem.bounds()
        assert np.all(xu >= xl)
        X = xl + (xu - xl) * X

    return X


class FloatRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.random((n_samples, problem.n_var))

        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X

        return X


class BinaryRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)


class IntegerRandomSampling(FloatRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        return np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)])


class PermutationRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), 0, dtype=int)
        for i in range(n_samples):
            X[i, :] = np.random.permutation(problem.n_var)
        return X
    
    
class VNFRandomSampling(Sampling):
    
    def _do(self, problem, n_samples, **kwargs):
        num_server = problem.num_server
        vnf_max = problem.vnf_max
        num_vnf = problem.num_vnf
        
        X = np.zeros((n_samples, num_server * vnf_max), dtype=int)
        assert len(X.shape) == 2, "Wrong dimension"
        for i in range(n_samples):
            indi = np.zeros((num_server * vnf_max,), dtype=int)
            values = np.arange(num_vnf + 1)
            visited = np.zeros((num_server * vnf_max,), dtype=bool)

            while not np.all(visited):
                for value in values:
                    unvisited_indices = np.where(visited == False)[0]
                    if len(unvisited_indices) == 0:
                        break
                    selected_index = np.random.choice(unvisited_indices)
                    indi[selected_index] = value
                    visited[selected_index] = True

            for ser in range(num_server):
                start = ser * vnf_max
                end = start + vnf_max - 1
                values_in_range = indi[start:end+1]
                unique_values, counts = np.unique(values_in_range, return_counts=True)
                repeated_values = unique_values[counts > 1]
                for value in repeated_values:
                    indices = np.where(values_in_range == value)[0]
                    if len(indices) >= 2:
                        indi[start + indices[1]] = num_vnf

            X[i] = indi
        return X
