"""Graph analysis utilities for the MAIS simulation.

This module provides helper functions for computing structural properties of
the contact graph used in the simulation, such as node degree statistics
derived from edge-probability data.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def compute_mean_degree(graph, nodes):
    """Compute the mean expected number of contacts (degree) for a set of nodes.
    
    For each pair of nodes in the graph, the function aggregates all edges
    connecting them to compute the probability that *at least one* contact
    occurs on any layer. This probability is stored in a dense sparse matrix,
    and the expected degree of each node in ``nodes`` is the row sum of that
    matrix. The mean is then taken over all nodes in ``nodes``.
    
    The computation iterates over all node pairs, so it is suited for analysis
    rather than performance-critical simulation paths.
    
    Args:
        graph (LightGraph): The contact graph object. Must expose:
            - ``graph.num_nodes`` (int): Total number of nodes.
            - ``graph.nodes`` (iterable): All node indices.
            - ``graph.A`` (array-like): Adjacency structure mapping a node
            pair ``(n1, n2)`` to an index into ``graph.edges_repo``
            (``0`` means no edge).
            - ``graph.edges_repo`` (list): Repository of edge collections
            keyed by the index returned from ``graph.A``.
            - ``graph.get_edges_probs(edges)`` (callable): Returns an array
            of transmission probabilities for the given edges.
        nodes (iterable): Subset of node indices for which the mean degree is
            computed.

    Returns:
        float: Mean expected number of contacts per node over the given
        ``nodes``.
    """

    # first create matrix of all probs of contacts
    # (can be optimised, I do not care about time now - so for all nodes)
    graph_matrix = lil_matrix((graph.num_nodes, graph.num_nodes), dtype=float)
    
    for n1 in graph.nodes:
        for n2 in graph.nodes:
            index = graph.A[n1, n2]
            if index == 0: # no edge
                continue
            edges_repo = graph.edges_repo[index]
            probs = graph.get_edges_probs(np.array(edges_repo))
            probs = 1 - probs
            graph_matrix[n1, n2] = 1 - probs.prod()
    graph_matrix = csr_matrix(graph_matrix)


    def node_degree(node):
        return graph_matrix[node].sum()

    degrees = [node_degree(node) for node in nodes]
    return sum(degrees)/len(degrees)

