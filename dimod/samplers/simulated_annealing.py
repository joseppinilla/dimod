import sys
import random
import math
import itertools
from multiprocessing import Pool

from dimod import DiscreteModelSampler
from dimod.decorators import ising, qubo
from dimod import ising_energy, qubo_to_ising, SpinResponse

__all__ = ['SimulatedAnnealingSampler']


if sys.version_info[0] == 2:
    range = xrange


class SimulatedAnnealingSampler(DiscreteModelSampler):

    @ising(1, 2)
    def sample_ising(self, h, J, beta_range=(.1, 3.33), n_samples=10, sweeps=1000,
                     multiprocessing=False):
        """Sample from low-energy spin states using simulated annealing.

        Args:
            h (dict): A dictionary of the linear biases in the Ising
            problem. Should be of the form {v: bias, ...} for each
            variable v in the Ising problem.
            J (dict): A dictionary of the quadratic biases in the Ising
            problem. Should be a dict of the form {(u, v): bias, ...}
            for each edge (u, v) in the Ising problem. If J[(u, v)] and
            J[(v, u)] exist then the biases are added.
            beta_range (tuple, optional): A 2-tuple defining the
            beginning and end of the beta schedule (beta is the
            inverse temperature). The schedule is applied linearly
            in beta. Default is (.1, 3.33).
            n_samples (int, optional): Each sample is the result of
            a single run of the simulated annealing algorithm.
            sweeps (int, optional): The number of sweeps or steps.
            Default is 1000.
            multiprocessing (bool, optional): When True, the simulated
            annealing algorithms are run in parallel using the Python
            multiprocessing library.

        Returns:
            :obj:`SpinResponse`

        Examples:
            >>> sampler = SimulatedAnnealingSampler()
            >>> h = {0: -1, 1: -1}
            >>> J = {(0, 1): -1}
            >>> response = sampler.sample_ising(h, J, samples=1)
            >>> list(response.samples())
            [{0: 1, 1: 1}]

        """

        # input checking
        # h, J are handled by the @ising decorator
        # beta_range, sweeps are handled by ising_simulated_annealing
        if not isinstance(n_samples, int):
            raise TypeError("'samples' should be a positive integer")
        if n_samples < 1:
            raise ValueError("'samples' should be a positive integer")

        # create the response object. Ising returns spin values.
        response = SpinResponse()

        # now we use ising_simulated_annealing to generate samples, either in
        # parallel or not.
        if not multiprocessing or n_samples < 2:
            # if the multiprocessing flag is False or the user is only requesting 1
            # sample then we can just run ising_simulated_annealing directly
            for __ in range(n_samples):
                sample, energy = ising_simulated_annealing(h, J, beta_range, sweeps)
                response.add_sample(sample, energy)

        else:
            # if the multiprocessing flag is set to true, we can run the
            # ising_simulated_annealing functions in parallel for each sample
            # because of the limitations of Pool.map, we need to give the arguments
            # as a single tuple.
            args = itertools.repeat((h, J, beta_range, sweeps), n_samples)
            for sample, energy in Pool(n_samples).map(_ising_simulated_annealing_single_arg, args):
                response.add_sample(sample, energy)

        return response

    def sample_structured_ising(self, h, J, **args):
        return self.sample_ising(h, J, **args)

    @qubo(1)
    def sample_qubo(self, Q, **args):
        h, J, offset = qubo_to_ising(Q)
        spin_response = self.sample_ising(h, J, **args)
        return spin_response.as_binary(offset)

    def sample_structured_qubo(self, Q, **args):
        return self.sample_qubo(Q, **args)


def _ising_simulated_annealing_single_arg(args):
    """Allows ising_simulated_annealing to be used with Pool.map"""
    return ising_simulated_annealing(*args)


def ising_simulated_annealing(h, J, beta_range=(.1, 3.33), sweeps=1000):
    """Tries to find the spins that minimize the given Ising problem.

    Args:
        h (dict): A dictionary of the linear biases in the Ising
        problem. Should be of the form {v: bias, ...} for each
        variable v in the Ising problem.
        J (dict): A dictionary of the quadratic biases in the Ising
        problem. Should be a dict of the form {(u, v): bias, ...}
        for each edge (u, v) in the Ising problem. If J[(u, v)] and
        J[(v, u)] exist then the biases are added.
        beta_range (tuple, optional): A 2-tuple defining the
        beginning and end of the beta schedule (beta is the
        inverse temperature). The schedule is applied linearly
        in beta. Default is (.1, 3.33).
        sweeps (int, optional): The number of sweeps or steps.
        Default is 1000.

    Returns:
        dict: A sample as a dictionary of spins.
        float: The energy of the returned sample.

    Raises:
        TypeError: If the values in `beta_range` are not numeric.
        TypeError: If `sweeps` is not an int.
        TypeError: If `beta_range` is not a tuple.
        ValueError: If the values in `beta_range` are not positive.
        ValueError: If `beta_range` is not a 2-tuple.
        ValueError: If `sweeps` is not positive.

    https://en.wikipedia.org/wiki/Simulated_annealing

    """

    # input checking, assume h and J are already checked
    if not isinstance(beta_range, (tuple, list)):
        raise TypeError("'beta_range' should be a tuple of length 2")
    if any(not isinstance(b, (int, float)) for b in beta_range):
        raise TypeError("values in 'beta_range' should be numeric")
    if any(b <= 0 for b in beta_range):
        raise ValueError("beta values in 'beta_range' should be positive")
    if len(beta_range) != 2:
        raise ValueError("'beta_range' should be a tuple of length 2")
    if not isinstance(sweeps, int):
        raise TypeError("'sweeps' should be a positive int")
    if sweeps <= 0:
        raise ValueError("'sweeps' should be a positive int")

    # We want the schedule to be linear in beta (inverse temperature)
    beta_init, beta_final = beta_range
    betas = [beta_init + i * (beta_final - beta_init) / (sweeps - 1.) for i in range(sweeps)]

    # set up the adjacency matrix. We can rely on every node in J already being in h
    adj = {n: set() for n in h}
    for n0, n1 in J:
        adj[n0].add(n1)
        adj[n1].add(n0)

    # we will use a vertex coloring the the graph and update the nodes by color. A quick
    # greedy coloring will be sufficient.
    __, colors = greedy_coloring(adj)

    # let's make our initial guess (randomly)
    spins = {v: random.choice((-1, 1)) for v in h}

    # there are exactly as many betas as sweeps
    for beta in betas:

        # we want to know the gain in energy for flipping each of the spins
        # we can calculate all of the linear terms simultaniously
        energy_diff_h = {v: -2 * spins[v] * h[v] for v in h}

        # for each color, do updates
        for color in colors:

            nodes = colors[color]

            # we now want to know the energy change for flipping the spins within
            # the color class
            energy_diff_J = {}
            for v0 in nodes:
                ediff = 0
                for v1 in adj[v0]:
                    if (v0, v1) in J:
                        ediff += spins[v0] * spins[v1] * J[(v0, v1)]
                    if (v1, v0) in J:
                        ediff += spins[v0] * spins[v1] * J[(v1, v0)]

                energy_diff_J[v0] = -2. * ediff

            # now decide whether to flip spins according to the
            # following scheme:
            #   p ~ Uniform(0, 1)
            #   log(p) < -beta * (energy_diff)
            for v in nodes:
                logp = math.log(random.uniform(0, 1))
                if logp < -1. * beta * (energy_diff_h[v] + energy_diff_J[v]):
                    # flip the variable in the spins
                    spins[v] *= -1

    return spins, ising_energy(h, J, spins)


def greedy_coloring(adj):
    """Determines a vertex coloring.

    Args:
        adj (dict): The edge structure of the graph to be colored.
        `adj` should be of the form {node: neighbors, ...} where
        neighbors is a set.

    Returns:
        dict: the coloring {node: color, ...}
        dict: the colors {color: [node, ...], ...}

    Note:
        This is a greedy heuristic, the resulting coloring is not
        necessarily minimal.

    """

    # now let's start coloring
    coloring = {}
    colors = {}
    possible_colors = {n: set(range(len(adj))) for n in adj}
    while possible_colors:

        # get the n with the fewest possible colors
        n = min(possible_colors, key=lambda n: len(possible_colors[n]))

        # assign that node the lowest color it can still have
        color = min(possible_colors[n])
        coloring[n] = color
        if color not in colors:
            colors[color] = {n}
        else:
            colors[color].add(n)

        # also remove color from the possible colors for n's neighbors
        for neighbor in adj[n]:
            if neighbor in possible_colors and color in possible_colors[neighbor]:
                possible_colors[neighbor].remove(color)

        # finally remove n from nodes
        del possible_colors[n]

    return coloring, colors