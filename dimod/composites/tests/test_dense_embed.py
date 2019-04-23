import unittest
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import timeit
from random import random
import dimod
from dimod.samplers.tests.generic_sampler_tests import SamplerAPITest
# For future integration or tests
# import dwave_sapi_dimod as sapi

class MockSampler(dimod.TemplateSampler):
    def sample_ising(self, h, J, orig_h=None):

        if orig_h is not None:
            assert h != orig_h
        return dimod.ExactSolver().sample_ising(h,J)
        #return sapi.SAPILocalSampler('c4-sw_optimize').sample_ising(h,J)


    def sample_qubo(self, Q):

        return dimod.ExactSolver().sample_qubo(Q)
        #return sapi.SAPILocalSampler('c4-sw_optimize').sample_qubo(Q)


class TestQCADenseEmbed(unittest.TestCase):

    def setUp(self):
        self.Ek_femag = -1
        self.Ek_anti = 0.217

    def test_qca_not(self):
        '''QCA NOT Gate graph

                3---5
              / | /  \
        D---0---2     6
              \ | \  /
                1---4

        '''
        Ek_anti = self.Ek_anti
        Ek_femag = self.Ek_femag

        Ek_femag = -1
        Ek_anti = 0.217


        J = { (0,1):Ek_anti, (0,3):Ek_anti,(0,2):Ek_femag,
                (1,2):Ek_femag,(1,4):Ek_femag,(2,3):Ek_femag,
                (2,4):Ek_anti,(2,5):Ek_anti,(3,5):Ek_femag,
                (4,6):Ek_anti,(5,6):Ek_anti}

        h = {0:1,1:0,2:0,3:0,4:0,5:0,6:0}

        m=4
        n=4
        G = dnx.chimera_graph(m,n)

        golden_pos = {0:1,      1:1,    2:1,    3:1,    4:1,    5:1,    6:-1}
        golden_neg = {0:-1,     1:-1,   2:-1,   3:-1,   4:-1,   5:-1,   6:1}

        #dnx.draw_chimera(G)
        sampler = MockSampler()
        # Force the structure on the MockSampler
        sampler.structure = (list(G.nodes()),list(G.edges()))
        composed_sampler = dimod.DenseEmbed(sampler)

        Q, offset = dimod.ising_to_qubo(h,J)
        response =  composed_sampler.sample_ising(h,J)

        self.assertTrue(response)
        response =  composed_sampler.sample_qubo(Q)
        self.assertTrue(response)

class TestDenseEmbed(unittest.TestCase):

    def test_k3(self):

        J = {(0, 1): -1, (1, 2): -1, (2, 0): -1}
        h = {0:-0.5,1:-1,2:0.0}


        m=4
        n=4
        #h = {k:random()*2-1 for k in range(m*n*8)}
        G = dnx.chimera_graph(m,n)

        #dnx.draw_chimera(G,h,J)
        sampler = MockSampler()
        # Force the structure on the MockSampler
        sampler.structure = (list(G.nodes()),list(G.edges()))
        composed_sampler = dimod.DenseEmbed(sampler)


        Q, offset = dimod.ising_to_qubo(h,J)
        response =  composed_sampler.sample_ising(h,J)


        self.assertTrue(response)
        response =  composed_sampler.sample_qubo(Q)
        self.assertTrue(response)
