"""Minimal directed graph for information-spread models.

This module provides :class:`SimpleGraph`, a lightweight directed graph
representation designed for information models (e.g. opinion dynamics,
rumour spreading) where edges are oriented and all transmission probabilities
are uniform.  The graph is stored as two NumPy arrays—one for edge sources
and one for edge destinations—keeping memory overhead minimal.
"""

from copy import copy
import logging
import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import csr_matrix, lil_matrix


class SimpleGraph:
    """Lightweight directed graph for information models.

    The graph stores only directed edges (no layer information, no
    per-edge probabilities) as two NumPy arrays of source and destination
    node indices.  This is sufficient for simple information/opinion models
    where all contacts are equally likely to transmit.

    Attributes:
        random_seed (int or None): Seed supplied at construction, stored for
            reproducibility bookkeeping.
        e_source (numpy.ndarray): Source node index for each edge.
        e_dest (numpy.ndarray): Destination node index for each edge.
        num_nodes (int): Total number of nodes inferred from the maximum
            node index seen in the edge list.

    Args:
        random_seed (int, optional): Seed for NumPy's random number
            generator.  ``None`` leaves the global RNG state unchanged.
    """

    def __init__(self, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        self.random_seed = random_seed

    def read_csv(self,
                 path_to_edges='edges.csv'):
        """Load a directed edge list from a CSV (or whitespace-separated) file.

        The file must contain exactly two columns—source node and destination
        node—either comma- or whitespace-separated, with no header row.
        Self-loops are dropped with a warning logged at the WARNING level.
        The node count is inferred as ``max(source, dest) + 1``.

        Args:
            path_to_edges (str): Path to the edge-list file.
                Defaults to ``'edges.csv'``.
        """
        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}
        edges = pd.read_csv(path_to_edges,
                            header=None,
                            sep=r",|\s+",
                            engine='python',
                            names=['vertex1', 'vertex2'],
                            **csv_hacking)
    

        # edges
        # drop self edges
        indexNames = edges[edges['vertex1'] == edges['vertex2']].index
        if len(indexNames):
            logging.warning("Warning: dropping self edges!!!!")
            edges.drop(indexNames, inplace=True)

        # fill edges to a graph
        n_edges = len(edges)
        # edges data"

        # self.e_source = np.empty(n_edges, dtype="uint32")
        # self.e_dest = np.empty(n_edges, dtype="uint32")

        # for i, row in tqdm.tqdm(enumerate(edges.itertuples()), total=len(edges)):
        #     self.e_source[i] = row.vertex1 # followee  
        #     self.e_dest[i] = row.vertex2   # follower

        self.e_source = edges["vertex1"].values
        self.e_dest = edges["vertex2"].values
            
        self.num_nodes = max(self.e_source.max(), self.e_dest.max()) + 1
        logging.info(f"Graph loaded with {n_edges} edges and {self.num_nodes} nodes.")
        
    @property
    def number_of_nodes(self):
        """Total number of nodes in the graph.

        Returns:
            int: Number of nodes (inferred from the maximum node index in
            the edge list, plus one).
        """
        return self.num_nodes


    def copy(self):
        """Return a shallow copy of the graph.

        Because :class:`SimpleGraph` has no mutable run-time state (unlike
        :class:`~graphs.light.LightGraph`) a plain shallow copy is sufficient
        to produce an independent graph object for a new simulation run.

        Returns:
            SimpleGraph: A shallow copy of ``self``.
        """
        return copy(self)

    
