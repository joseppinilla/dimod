# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
"""
A solver that calculates the energy of all possible samples.

Note:
    This sampler is designed for use in testing. Because it calculates the
    energy for every possible sample, it is very slow.

"""
import itertools

import numpy as np
from six.moves import zip

from dimod.core.sampler import Sampler
from dimod.decorators import bqm_index_labels
from dimod.sampleset import SampleSet
from dimod.vartypes import Vartype

__all__ = ['ExactSolver', 'ExactDeltaSolver']

class ExactSolver(Sampler):
    """A simple exact solver for testing and debugging code using your local CPU.

    Notes:
        This solver becomes slow for problems with 18 or more
        variables.

    Examples:
        This example solves a two-variable Ising model.

        >>> h = {'a': -0.5, 'b': 1.0}
        >>> J = {('a', 'b'): -1.5}
        >>> sampleset = dimod.ExactSolver().sample_ising(h, J)
        >>> print(sampleset)
           a  b energy num_oc.
        0 -1 -1   -2.0       1
        2 +1 +1   -1.0       1
        1 +1 -1    0.0       1
        3 -1 +1    3.0       1
        ['SPIN', 4 rows, 4 samples, 2 variables]

        This example solves a two-variable QUBO.

        >>> Q = {('a', 'b'): 2.0, ('a', 'a'): 1.0, ('b', 'b'): -0.5}
        >>> sampleset = dimod.ExactSolver().sample_qubo(Q)


        This example solves a two-variable binary quadratic model

        >>> bqm = dimod.BinaryQuadraticModel({'a': 1.5}, {('a', 'b'): -1}, 0.0, 'SPIN')
        >>> sampleset = dimod.ExactSolver().sample(bqm)

    """
    properties = None
    parameters = None

    def __init__(self):
        self.properties = {}
        self.parameters = {}

    @bqm_index_labels
    def sample(self, bqm):
        """Sample from a binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

        Returns:
            :obj:`~dimod.SampleSet`

        """
        M = bqm.binary.to_numpy_matrix()
        off = bqm.binary.offset

        if M.shape == (0, 0):
            return SampleSet.from_samples([], bqm.vartype, energy=[])

        sample = np.zeros((len(bqm),), dtype=bool)

        # now we iterate, flipping one bit at a time until we have
        # traversed all samples. This is a Gray code.
        # https://en.wikipedia.org/wiki/Gray_code
        def iter_samples():
            sample = np.zeros((len(bqm)), dtype=bool)
            energy = 0.0

            yield sample.copy(), energy + off

            for i in range(1, 1 << len(bqm)):
                v = _ffs(i)

                # flip the bit in the sample
                sample[v] = not sample[v]

                # for now just calculate the energy, but there is a more clever way by calculating
                # the energy delta for the single bit flip, don't have time, pull requests
                # appreciated!
                energy = sample.dot(M).dot(sample.transpose())

                yield sample.copy(), float(energy) + off

        samples, energies = zip(*iter_samples())

        response = SampleSet.from_samples(np.array(samples, dtype='int8'), Vartype.BINARY, energies)

        # make sure the response matches the given vartype, in-place.
        response.change_vartype(bqm.vartype, inplace=True)

        return response

class DynamicDeltaSolver(Sampler):
    def __init__(self):
        self.properties = {}
        self.parameters = {}

    @bqm_index_labels
    def sample(self, bqm):
        """Sample from a binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

        Returns:
            :obj:`~dimod.Response`: A `dimod` :obj:`.~dimod.Response` object.

        """
        M = bqm.binary.to_numpy_matrix()
        off = bqm.binary.offset

        if M.shape == (0, 0):
            return Response.from_samples([], {'energy': []}, {}, bqm.vartype)

        vartype = bool

        sample = np.zeros((len(bqm)), dtype=vartype)

        info = {}

        reponse = DynamicResponse(variables, info, vartype)


        return response

class ExactDeltaSolver(Sampler):
    """An exact solver using energy deltas for testing and debugging.
    """
    properties = None
    parameters = None

    def __init__(self):
        self.properties = {}
        self.parameters = {}

    @bqm_index_labels
    def sample(self, bqm):
        """Sample from a binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

        Returns:
            :obj:`~dimod.Response`: A `dimod` :obj:`.~dimod.Response` object.

        """
        M = bqm.binary.to_numpy_matrix()
        off = bqm.binary.offset

        if M.shape == (0, 0):
            return SampleSet.from_samples([], bqm.vartype, energy=[])

        sample = np.zeros((len(bqm)), dtype=bool)

        # now we iterate, flipping one bit at a time until we have
        # traversed all samples. This is a Gray code.
        # https://en.wikipedia.org/wiki/Gray_code
        def iter_samples():

            # energy = 0.0
            energy = off
            yield sample.copy(), energy

            for i in range(1, 1 << len(bqm)):
                v = _ffs(i)

                # flip the bit in the sample
                sample[v] = not sample[v]

                # calculate energy delta from triu indices of bit flip
                # only upper triangular part of the matrix
                delta_e = 0.0
                for u in range(0, v):
                    J = M[u][v]
                    delta_e += J*sample[u]
                delta_e += M[v][v]
                for u in range(v+1,len(bqm)):
                    J = M[v][u]
                    delta_e += J*sample[u]

                # apply delta
                if sample[v]:
                    energy += delta_e
                else:
                    energy -= delta_e

                yield sample.copy(), float(energy)

        samples, energies = zip(*iter_samples())

        response = SampleSet.from_samples(np.array(samples, dtype='int8'), Vartype.BINARY, energies)

        # make sure the response matches the given vartype, in-place.
        response.change_vartype(bqm.vartype, inplace=True)

        return response

def _ffs(x):
    """Gets the index of the least significant set bit of x."""
    return (x & -x).bit_length() - 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import random
    import dimod

    # Problem parameters
    J_RANGE = [-2.0, 2.0]
    H_RANGE = [-1.0, 1.0]
    size = 18

    # Create Problem
    Sg = nx.complete_graph(size)
    for (u,v,data) in Sg.edges(data=True):
        data['weight'] = random.uniform(*J_RANGE)
    h = {v:random.uniform(*H_RANGE) for v in Sg}
    J = {(u,v):data['weight'] for u,v,data  in Sg.edges(data=True)}
    bqm = dimod.BinaryQuadraticModel(h, J, -0.5, dimod.SPIN)

    delta_sampler = ExactDeltaSolver()
    dimod_sampler = ExactSolver()


    %time delta_response = delta_sampler.sample(bqm)
    %time dimod_response = dimod_sampler.sample(bqm)

    # list(delta_response.data(['energy']))
    # list(dimod_response.data(['energy']))
