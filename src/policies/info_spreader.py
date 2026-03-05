"""Information-spreader seeding policy for the MAIS simulation.

This module defines the :class:`Spreader` policy, which uses PageRank
centrality to seed a highly-connected node into state ``I`` at the
start of the simulation.  It is intended for use with information-
diffusion models rather than epidemic models.
"""

import numpy as np
import graph_tool.all as gt
import logging


from policies.policy import Policy
from models.agent_info_models import STATES

class Spreader(Policy):

    """Policy that seeds the most central node into the infectious state.

    On the first simulation day, a ``graph-tool`` graph is built from
    the contact network, PageRank centrality is computed (weighted by
    edge probability, intensity, and layer weight), and the node at the
    requested quantile of centrality is moved to state ``I``.

    Args:
        graph: The contact network graph object.  Must expose
            ``e_source``, ``e_dest``, ``e_probs``, ``e_intensities``,
            ``e_types``, ``layer_weights``, and ``num_nodes``.
        model: The epidemic model instance (must support
            ``change_states`` and ``nodes``).
        quantile (float): Quantile in [0, 1] of the centrality
            distribution used to select the seed node.
            Defaults to 0.9.
    """

    def __init__(self, graph, model, quantile=0.9):
        """Initialise the spreader policy.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
            quantile (float): Centrality quantile for seed node
                selection.  Defaults to 0.9.
        """
        super().__init__(graph, model)
        self.quantile = quantile

    def first_day_setup(self):
        """Build the weighted graph, compute PageRank, and seed the central node.

        Constructs an undirected ``graph-tool`` graph with edge weights
        equal to ``prob * intensity * layer_weight``, runs PageRank,
        and changes the state of the node at ``self.quantile`` of the
        centrality distribution to ``STATES.I``.
        """
        print("Spreader policy: first day setup")

        # create graph_tool graph
        g = gt.Graph(directed=False)
        e_weight = g.new_edge_property("float")
        g.add_vertex(self.graph.num_nodes)
        for i, (source, dest) in enumerate(zip(self.graph.e_source, self.graph.e_dest)):
            e = g.add_edge(source, dest)
            e_weight[e] = self.graph.e_probs[i] * self.graph.e_intensities[i] * self.graph.layer_weights[self.graph.e_types[i]]

        # get nodes centralities 
        nodes_centralities = gt.pagerank(g, weight=e_weight)
        nodes_centralities = nodes_centralities.get_array()
        print(nodes_centralities.shape)
        print(len(nodes_centralities))
        indexes = sorted(range(len(nodes_centralities)), key=lambda k: nodes_centralities[k])        
        idx = indexes[int(len(nodes_centralities) * self.quantile) - 1]

        logging.info(f"Spreader policy: first day setup, node {idx} is selected with centrality {nodes_centralities[idx]}")
        
        self.model.change_states(self.model.nodes == idx, target_state=STATES.I)
 
    
    