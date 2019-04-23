'''
Created on Nov 23, 2017

@author: JosePinilla
'''
import time
import math
import traceback
import itertools
import networkx as nx
from random import random

import dwave_networkx as dnx

from dwave_embedding_utilities import  embed_ising, unembed_samples
#from dwave_sapi2.embedding import embed_problem, unembed_answer,
# from dwave_sapi2.embedding import _verify_embedding_1, _verify_embedding_2

from dimod import _PY2
from dimod.utilities import qubo_to_ising
from dimod.composites.template_composite import TemplateComposite
from dimod.decorators import ising, qubo
from dimod.responses import SpinResponse

try:
    from dimod.composites.dense_embed_core.embed import denseEmbed, setChimera
    from dimod.composites.dense_embed_core.convert import convertToModels
    from dimod.composites.dense_embed_core.utilities import linear_to_tuple, tuple_to_linear
except Exception as e:
    print('Could not load dense embedding method...')
    print (traceback.print_exc())

__all__ = ['DenseEmbed']

if _PY2:
    iteritems = lambda d: d.iteritems()
    range = xrange
    zip = itertools.izip
else:
    iteritems = lambda d: d.items()

DENSE_TRIALS = 10

class DenseEmbed(TemplateComposite):
    """Composite for dense embedding of an Ising problem into a Chimera
    Graph.

    Args:
        sampler: A structured dimod sampler object.


    """
    def __init__(self, sampler
                ):
        TemplateComposite.__init__(self, sampler)
        # copy over the sampler
        self._child = sampler
        # copy over the structure
        self.structure = sampler.structure

        #TODO: Implement cached_embeddings?


    def embed(self, qca_adj, chimera_adj):
        '''Setup and run the Dense Placement algorithm'''

        m = self.m
        n = self.n
        t = self.t

        # format embedding parameters
        setChimera(chimera_adj, m, n, t)

        # run a number of embedding and choose the best
        embeds = []
        for trial in xrange(DENSE_TRIALS):
            print('Trial {0}...'.format(trial)),
            try:
                cell_map, paths = denseEmbed(qca_adj, write=False)
                print('success')
            except Exception as e:
                print('failed')
                print (traceback.print_exc())
                continue
            embeds.append((cell_map, paths))

        if len(embeds) == 0:
            raise Exception('No embedding found')

        # sort embedding by number of qubits used (total path length)
        cell_map, paths = sorted(embeds, key=lambda x: sum([len(p) for p in x[1]]))[0]

        # get cell models
        print('Converting to models...')
        models, max_model = convertToModels(paths, cell_map)
        print('done')

        embedding = {}
        utilization = set()
        for k in models:
            qubits = [tuple_to_linear(tup=v,M=m,N=n,index0=True) for v in models[k]['qbits']]
            embedding[k] = qubits
            utilization.update(qubits)

        return embedding, utilization

    def parameter_setting(self, h, J, embedding, edgeset, utilization, chain_strength):

        h_emb={}

        #h0, J0, Jc = embed_problem(h, J, embedding, edgeset) #DWAVE SAPI2
        h0, J0, Jc = embed_ising(h, J, embedding, edgeset, chain_strength)

        # Filter linear biases to only qubits in embedding
        for q in utilization:
            h_emb.update({q:h0[q]})

        # Join quadratic and chain biases
        J0.update(Jc)
        # return the combination of target and chain quadradict values
        return h_emb, J0

    @ising(1, 2)
    def sample_ising(self, h, J,
                    spacing = 0.0,
                    chain_strength = 1.0,
                    t = 4,
                     **kwargs):
        """Minor embedding of an Ising problem into a Chimera graph,
        then samples the problem using the child sampler's
        `sample_ising` method.

        Args:
            h (dict/list): The linear terms in the Ising problem. If a
                dict, should be of the form {v: bias, ...} where v is
                a variable in the Ising problem, and bias is the linear
                bias associated with v. If a list, should be of the form
                [bias, ...] where the indices of the biases are the
                variables in the Ising problem.
            J (dict): A dictionary of the quadratic terms in the Ising
                problem. Should be of the form {(u, v): bias} where u,
                v are variables in the Ising problem and bias is the
                quadratic bias associated with u, v.
            spacing:
            chain_strength:
            t: Number of qubits per shore in the chimera graph.
            **kwargs: Any other keyword arguments are passed unchanged to
                the child sampler's `sample_ising` method.

        Returns:
            :obj:`SpinResponse`

        """
        sampler = self._child
        # Get the structure of the child sampler.
        (nodeset, edgeset) = sampler.structure

        # create the graph from the nodes and edges of the structured solver
        m = n = int(( math.sqrt( (len(nodeset)/(t*2))) ))
        chimera = dnx.chimera_graph(m,n,t,None,nodeset,edgeset)
        self.m,self.n,self.t = m,n,t

        chimera_adj_linear = {k : [i for i in chimera.adj[k].keys()] for k in nodeset}
        chimera_adj = {linear_to_tuple(k,m,n,t,True): [linear_to_tuple(i,m,n,t,True) for i in chimera.adj[k].keys()] for k in nodeset}

        chimera_adj_tuples = set( (u,v) for u in chimera_adj_linear for v in chimera_adj_linear[u] ) | set( (u,u) for u in chimera_adj_linear )

        qca = nx.Graph(list(J.keys()))

        qca_adj = {u:qca[u] for u in qca}

        embedding, utilization = self.embed(qca_adj,chimera_adj)

        # _verify_embedding_1(embedding.values(), chimera_adj_tuples)
        # _verify_embedding_2(embedding.values(), J, chimera_adj_tuples)

        # Parameter Setting
        h_embed, J_embed = self.parameter_setting(h, J, embedding, chimera_adj_linear, utilization, chain_strength)

        # invoke the child sampler
        emb_response = sampler.sample_ising(h_embed, J_embed, **kwargs)

        # unemnbed
        solutions = unembed_samples(emb_response, embedding, linear=h, quadratic=J)
        #solutions = unembed_answer(answers, embedding, 'minimize_energy', h, J)

        # and back once again into dicts for dimod...
        samples = ({v: sample[v] for v in h} for sample in solutions)
        sample_data = (data for __, data in emb_response.samples(data=True))
        response = SpinResponse()
        response.add_samples_from(samples, sample_data=sample_data, h=h, J=J)

        return response

    @qubo(1)
    def sample_qubo(self, Q,
                    spacing = 0.0,
                    chain_strength = 1.0,
                    t = 4,
                    **kwargs):
        """Minor embedding of a QUBO problem into a Chimera graph,
        then samples the problem using using the child sampler's
        `sample_ising` method.

        Args:
            Q: A dict of the QUBO coefficients.
            attr_dict (dict): A dictionary of the attributes of the nodes
                in the Ising proble. Layout-embeddeding uses the 'x' and 'y'
                attributes of the nodes to guide the embedding algorithm.
            **kwargs: Any other keyword arguments are passed unchanged to
                the child sampler's `sample_ising` method.
        """
        sampler =  self._child

        h, J, offset = qubo_to_ising(Q)

        ising_response = self.sample_ising(h, J, spacing=spacing, **kwargs)

        #TODO: offset transformation

        return ising_response
