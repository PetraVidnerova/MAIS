from copy import copy
import logging
import numpy as np
import pandas as pd
import tqdm 
from scipy.sparse import csr_matrix, lil_matrix


class SimpleGraph:

    """
    Graph for mainly for information models, 
    especially for simple ones.
    Graph has only edges, that are oriented.
    Probabilities of edges are uniform. 
    """

    def __init__(self, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        self.random_seed = random_seed

    def read_csv(self,
                 path_to_edges='edges.csv'):

        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}        
        edges = pd.read_csv(path_to_edges, **csv_hacking)
    
        
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
        return self.num_nodes


    def copy(self):
        """
        Optimized version of shallow/deepcopy of self.
        Since most fields never change between runs, we do shallow copies on them.
        :return: Shallow/deep copy of self.

        --> not needed for simple graph, use simple shallow copy
        """
        return copy(self)

    
