# these graphs are not used and the file is not maintained at the moment
# random graphs should be used in the future

# -*- coding: utf-8 -*-
"""NetworkX-based multi-layer graph generators (legacy, not actively maintained).

This module provides :class:`GraphGenerator` and several concrete subclasses
that build multi-layer contact graphs backed by a NetworkX ``MultiGraph``.
These classes pre-date the :class:`~graphs.light.LightGraph` implementation
and are no longer used in production simulations; random-graph variants are
planned for future use.

Classes:
    GraphGenerator: Base class wrapping a NetworkX MultiGraph with
    layer-aware helper methods.
    RandomSingleGraphGenerator: Generates a single-layer Barabasi-Albert
    graph with exponentially trimmed degree distribution.
    RandomGraphGenerator: Generates one Barabasi-Albert graph per layer
    with random truncated-normal edge weights.
    PickleGraphGenerator: Restores a graph from a NetworkX pickle file.
    CSVGraphGenerator: Builds a graph from nodes, edges and layer CSV files.
"""

import networkx as nx
import numpy as np
import scipy.stats as stats
from scipy.sparse import csr_matrix
import pandas as pd

from utils.sparse_utils import multiply_zeros_as_ones


class GraphGenerator:
    """Base class for multi-layer contact-graph generators.

    Wraps a NetworkX ``MultiGraph`` (``self.G``) and provides layer-aware
    helpers for building, querying and modifying multi-layer contact networks.
    Subclasses are responsible for populating ``self.G`` with nodes and edges.

    Class Attributes:
        layer_names (list[str]): Ordered list of layer name strings.
            Default covers 14 generic layers (``'F'`` through ``'Z'``).
        layer_probs (list[float]): Per-layer transmission probabilities,
            initialised to ``1.0`` for every layer.
        layer_types (list[int]): Integer indices corresponding to each layer.

    Attributes:
        G (networkx.MultiGraph): The underlying multi-layer graph.
        Graphs (dict): Optional per-layer single-layer graph cache populated
            by :meth:`as_dict_of_graphs`.
        A (scipy.sparse.csr_matrix or None): Cached final adjacency matrix.
            ``None`` until first computed.
        A_valid (bool): ``True`` when the cached adjacency matrix is up to
            date with the current edge weights.
        A_invalids (set): Set of ``(u, v)`` node pairs whose adjacency
            matrix entries need recomputing.
        quarantined_edges (dict): Mapping from edge keys to their original
            weights, used to restore edges after a quarantine is lifted.

    Args:
        random_seed (int, optional): Seed for NumPy's random number
            generator.  ``None`` leaves the global RNG state unchanged.
    """

    layer_names = ['F', 'D', 'P', 'E', 'H', 'K',
                   'C', 'S', 'O', 'L', 'R', 'T', 'X', 'Z']
    layer_probs = [1.0] * len(layer_names)
    layer_types = list(range(len(layer_names)))

    def __init__(self, random_seed=None):
        self.G = nx.MultiGraph()
        self.Graphs = {}
        self.G.graph['layer_names'] = self.layer_names
        self.G.graph['layer_probs'] = self.layer_probs

        if random_seed:
            np.random.seed(random_seed)

        self.A = None
        self.A_valid = False
        self.A_invalids = set()
        self.quarantined_edges = {}

    @property
    def nodes(self):
        """NodeView of all nodes in the underlying MultiGraph.

        Returns:
            networkx.classes.reportviews.NodeView: All nodes of ``self.G``.
        """
        return self.G.nodes

    def number_of_nodes(self):
        """Return the total number of nodes in the graph.

        Returns:
            int: Number of nodes in ``self.G``.
        """
        return self.G.number_of_nodes()

    def as_multigraph(self):
        """Return the raw underlying NetworkX MultiGraph.

        Returns:
            networkx.MultiGraph: The underlying graph object ``self.G``.
        """
        return self.G

    def as_one_graph(self):
        """Collapse the MultiGraph into a simple undirected Graph.

        Parallel edges between any pair of nodes are merged.

        Returns:
            networkx.Graph: A simple graph derived from ``self.G``.
        """
        return nx.Graph(self.G)

    def as_aggregated_graph(self):
        """Build an aggregated single-layer graph (incomplete, legacy stub).

        This method is a historical stub and does not return a fully
        populated graph.  It exists for backward-compatibility only.

        Returns:
            networkx.Graph: An empty graph with the same node set as
            ``self.G``.
        """
        ag = nx.Graph()
        ag.add_nodes_from(self.G)

        lp = []
        selected_edges = []

        for i, l in enumerate(self.layer_names):
            lp[i] = self.G.graph['layer_probs'][i]
            selected_edges[i] = [(u, v, e) for u, v, e in self.G.edges(
                data=True) if e['type'] == l]

        return ag

    def final_adjacency_matrix(self):
        """Compute (or return cached) the final contact-probability adjacency matrix.

        The matrix entry ``A[i, j]`` holds the probability that nodes ``i``
        and ``j`` make contact across *any* layer, calculated as
        ``1 - product_over_layers(1 - p_layer * w_edge)``.

        A two-level caching strategy is used:

        * If ``A_valid`` is ``True`` the cached matrix is returned
          immediately.
        * If the matrix has never been computed (``A is None``) or
          ``A_invalids`` is empty, the entire matrix is recalculated from
          scratch.
        * Otherwise only the entries listed in ``A_invalids`` are
          recomputed.

        Returns:
            scipy.sparse.csr_matrix: Symmetric probability matrix of shape
            ``(N, N)`` where ``N`` is the number of nodes.
        """

        if self.A_valid:
            return self.A

        if self.A is None or not self.A_invalids:
            N = self.number_of_nodes()
            prob_no_contact = csr_matrix((N, N))  # empty values = 1.0

            for name, prob in self.get_layers_info().items():
                a = nx.adj_matrix(self.get_graph_for_layer(name))
                if len(a.data) == 0:  # no edges, prob of no-contact = 1
                    continue
                a = a.multiply(prob)  # contact on layer
                not_a = a  # empty values = 1.0
                not_a.data = 1.0 - not_a.data  # prob of no-contace
                prob_no_contact = multiply_zeros_as_ones(
                    prob_no_contact, not_a)
                del a

            # prob_of_contact = 1.0 - prob_no_contact
            prob_of_contact = prob_no_contact
            prob_of_contact.data = 1.0 - prob_no_contact.data
            self.A = prob_of_contact
            self.A_valid = True
            self.A_invalids = set()
            return self.A

        else:
            # recalculate only invalids
            for u, v in list(self.A_invalids):
                edges_to_aggregate = self.G[u][v]
                sublayers = dict()
                for e in edges_to_aggregate.values():
                    sublayers[e["type"]] = sublayers.get(
                        e["type"], 0) + e["weight"]

                no_contact_prob = 1.0
                for layer_type, weight in sublayers.items():
                    w = 1.0 - weight * self.G.graph["layer_probs"][layer_type]
                    no_contact_prob *= w
                self.A[u, v] = 1.0 - no_contact_prob
                self.A[v, u] = 1.0 - no_contact_prob
            self.A_valid = True
            self.A_invalids = set()
            return self.A

    def get_graph_for_layer(self, layer_name):
        """Extract a single-layer subgraph for the named layer.

        Args:
            layer_name (str): Name of the layer to extract, must appear in
                ``self.layer_names``.

        Returns:
            networkx.Graph: A new simple graph containing all nodes of
            ``self.G`` and only the edges whose ``'type'`` attribute matches
            the index of ``layer_name`` in ``self.layer_names``.  The graph's
            metadata includes ``'layer_name'`` and ``'layer_prob'``.
        """
        ret_g = nx.Graph()
        ret_g.graph['layer_name'] = layer_name
        layer_index = self.layer_names.index(layer_name)
        ret_g.graph['layer_prob'] = self.G.graph['layer_probs'][layer_index]

        ret_g.add_nodes_from(self.G)
        selected_edges = [(u, v, e)
                          for u, v, e in self.G.edges(data=True)
                          if e['type'] == self.layer_names.index(layer_name)]
        ret_g.add_edges_from(selected_edges)

        return ret_g

    def as_dict_of_graphs(self):
        """Build a per-layer dictionary of single-layer graphs.

        Iterates over all layers, constructs a separate :class:`networkx.Graph`
        for each one (all nodes, only edges on that layer) and caches the
        result in ``self.Graphs``.

        Returns:
            dict[str, networkx.Graph]: Mapping of layer name to its
            corresponding single-layer graph.
        """
        self.Graphs = {}

        for i, l in enumerate(self.layer_names):
            FG = nx.Graph()
            FG.graph['layer_name'] = l
            FG.graph['layer_prob'] = self.G.graph['layer_probs'][i]
            FG.add_nodes_from(self.G)
            selected_edges = [(u, v, e) for u, v, e in self.G.edges(
                data=True) if e['type'] == l]
            FG.add_edges_from(selected_edges)
            self.Graphs[l] = FG
        return self.Graphs

    def get_attr_list(self, attr):
        """Return a list of node attribute values in node-iteration order.

        Args:
            attr (str): Name of the node attribute to collect.

        Returns:
            list: Attribute values for every node in ``self.G``, ordered by
            NetworkX's default node-iteration order.
        """
        attr_list = []
        for (p, d) in self.G.nodes(data=True):
            attr_list.append(d[attr])
        return attr_list

    def get_edges_for_node(self, node_id):
        """Return all edges incident to a given node.

        Args:
            node_id: Node identifier as stored in ``self.G``.  If the node
                does not exist a warning is printed and ``None`` is returned
                implicitly.

        Returns:
            networkx.classes.reportviews.OutMultiEdgeDataView or None:
            An edge-data view for all edges incident to ``node_id``, or
            ``None`` (implicit) if the node is not in the graph.
        """
        if not self.G.has_node(node_id):
            print(f"Warning: modify_layer_for_node called for nonexisting node_id {node_id}")
            return

        # for key in what_by_what:
        #     edges_to_modify = [
        #         (u, v, k) for u, v, k, d in self.G.edges(node, data=True, keys=True)
        #         if d['label'] == key
        #     ]
        #     print(edges_to_modify)
        #     for e in edges_to_modify:
        #         self.G.edges[e]['weight'] *= what_by_what[key]

        # for e in self.G.edges(node, data=True, keys=True):
        #     print(*e)

#        for e in self.G.edges([node_id], data=True, keys=True):
#            print(*e)
        return self.G.edges([node_id], data=True, keys=True)

    def get_layers_for_edge(self, u, v):
        """Return the list of layer types for all parallel edges between two nodes.

        Args:
            u: Source node identifier.
            v: Destination node identifier.

        Returns:
            list[int]: Layer type (``'type'`` attribute) for each parallel
            edge between ``u`` and ``v`` in the MultiGraph.
        """
        edges = self.G[u][v]
        return [d["type"] for d in edges.values()]

    def modify_layers_for_nodes(self, node_id_list, what_by_what, is_quarrantined=None):
        """Apply quarantine by scaling edge weights for given nodes.

        For each edge incident to a node in ``node_id_list`` whose layer
        appears in ``what_by_what``, the edge weight is multiplied by the
        corresponding coefficient (clipped to ``[0, 1]``).  Edges that are
        already quarantined (i.e. at least one endpoint has a non-zero
        quarantine counter in ``is_quarrantined``) are skipped.

        The original weight is recorded in ``self.quarantined_edges`` so it
        can be restored by :meth:`recover_edges_for_nodes`.  Affected
        ``(u, v)`` pairs are added to ``A_invalids`` so the adjacency matrix
        will be partially recomputed on next access.

        Args:
            node_id_list (iterable): Node identifiers whose incident edges
                should be modified.
            what_by_what (dict): Mapping of ``{layer_type: coefficient}``
                specifying the weight multiplier for each layer.  If falsy
                (empty or ``None``) the method returns immediately.
            is_quarrantined (numpy.ndarray, optional): Per-node quarantine
                counters.  When provided, edges where either endpoint has a
                non-zero counter are skipped.
        """

        if not what_by_what:
            return

        # if not self.G.has_node(node_id):
        #     print(f"Warning: modify_layer_for_node called for nonexisting node_id {node_id}")
        #     return

        changed = set()
        # keep the original list (it is modified in the cycle)
        for u, v, k, d in self.G.edges(set(node_id_list), data=True, keys=True):
            if is_quarrantined is not None and (is_quarrantined[u] > 0 or is_quarrantined[v] > 0):
                # edge is already quarrantined
                continue
            layer_label = d["type"]
            if layer_label in what_by_what:
                assert (u, v, k, d["type"], d["subtype"]) not in self.quarantined_edges, \
                    f"edge {(u, v, k, d['type'], d['subtype'])} is already quaratined, {is_quarrantined[u]}{is_quarrantined[v]}"
                s, e = (u, v) if u < v else (v, u)
                self.quarantined_edges[(
                    s, e, k, d["type"], d["subtype"])] = d['weight']
                self.G.edges[(u, v, k)
                             ]['weight'] = min(self.G.edges[(u, v, k)]['weight'] * what_by_what[layer_label], 1.0)
                changed.add((u, v))

        if changed:
            self.A_invalids.update(changed)
            self.A_valid = False

    def recover_edges_for_nodes(self, release, normal_life, is_quarrantined):
        """Restore original edge weights for nodes being released from quarantine.

        Iterates over all edges incident to ``release`` nodes.  For edges
        where both endpoints have a zero quarantine counter in
        ``is_quarrantined``, the weight stored in ``self.quarantined_edges``
        is written back.  Edges where at least one endpoint is still
        quarantined are left unchanged.

        Affected ``(u, v)`` pairs are added to ``A_invalids`` so the
        adjacency matrix is recomputed on next access.

        Args:
            release (iterable): Node identifiers being released from
                quarantine.
            normal_life: Unused parameter retained for interface
                compatibility with the ``LightGraph`` API.
            is_quarrantined (numpy.ndarray): Per-node quarantine counters
                indexed by node ID.  Nodes with a counter of ``0`` are
                considered free.
        """
        changed = set()
        for u, v, k, d in self.G.edges(release, data=True, keys=True):
            if is_quarrantined[u] or is_quarrantined[v]:
                # still one of nodes is in quarrantine
                continue
            e_type = d["type"]
            e_subtype = d["subtype"]
            # new_weight = None
            # for edata in normal_life.G.get_edge_data(u, v).values():
            #     if edata["type"] == e_type and edata["subtype"] == e_subtype:
            #         new_weight = edata["weight"]
            #         break
            # if new_weight is None:
            #     raise ValueError("Edge not found in normal life.")
            s, e = (u, v) if u < v else (v, u)
            self.G.edges[(u, v, k)]['weight'] = self.quarantined_edges[(
                s, e, k, e_type, e_subtype)]
            del self.quarantined_edges[(
                s, e, k, e_type, e_subtype)]
            changed.add((s, e))
        if changed:
            self.A_invalids.update(changed)
            self.A_valid = False

    def get_layers_info(self):
        """Return a mapping of layer names to their transmission probabilities.

        Returns:
            dict[str, float]: Dictionary mapping each layer name to its
            current transmission probability as stored in
            ``self.G.graph['layer_probs']``.
        """
        return dict(zip(self.G.graph["layer_names"], self.G.graph["layer_probs"]))

    def print_multi(self):
        """Print the MultiGraph in DOT format to stdout."""
        dot_G = nx.nx_pydot.to_pydot(self.G)
        print(dot_G)

    def draw_multi(self, filename='empty_graph.png'):
        """Render the MultiGraph using Graphviz and save it to a file.

        Args:
            filename (str): Output file path for the rendered image.
                Defaults to ``'empty_graph.png'``.
        """
        A = nx.nx_agraph.to_agraph(self.G)
        A.layout('dot')
        A.draw(filename)

    def close_layers(self, list_of_layers, coefs=None):
        """Set transmission probability to zero (or a custom value) for named layers.

        Invalidates the cached adjacency matrix so it will be recomputed on
        next access.

        Args:
            list_of_layers (list[str]): Names of layers to close.
            coefs (list[float], optional): Per-layer replacement probabilities.
                When provided, ``coefs[idx]`` is used for the layer at
                position ``idx`` in ``list_of_layers``.  When ``None`` all
                listed layers are set to ``0``.
        """
        for idx, name in enumerate(list_of_layers):
            i = self.G.graph["layer_names"].index(name)
            self.G.graph["layer_probs"][i] = 0 if not coefs else coefs[idx]
            print(f"Closing {name} (new value {self.G.graph['layer_probs'][i]})")
        self.A_valid = False

    def write_pickle(self, path):
        """Serialize the underlying NetworkX graph to a pickle file.

        Args:
            path (str): Destination file path for the pickle.
        """
        nx.write_gpickle(self.G, path)

    def read_picke(self, path):
        """Restore the graph from a NetworkX pickle file.

        Note: The method name contains a deliberate historical typo
        (``read_picke`` instead of ``read_pickle``); it is kept as-is for
        backward compatibility.

        Args:
            path (str): Path to the pickle file to read.
        """
        self.G = nx.read_gpickle(path)
        self.layer_names = self.G.graph['layer_names']
        self.layer_probs = self.G.graph['layer_probs']


def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    """Generate a power-law-esque graph by thinning a Barabasi-Albert base graph.

    Starts from a Barabasi-Albert preferential-attachment graph (expected to
    form a single connected component) and then stochastically removes edges
    from each node so that the resulting degree distribution has an
    exponential tail rather than a hard minimum of ``m``.

    For each node the number of edges to *keep* is drawn from an exponential
    distribution with the given ``scale``, clipped to
    ``[min_num_edges, current_degree]``.  Edges not selected for retention
    are removed.

    Args:
        base_graph (networkx.Graph, optional): Pre-existing graph to thin.
            When ``None`` a fresh Barabasi-Albert graph of ``n`` nodes is
            created.
        scale (float): Scale parameter of the exponential distribution used
            to sample the number of edges to keep per node.  Defaults to
            ``100``.
        min_num_edges (int): Minimum number of edges guaranteed to be kept
            per node.  Defaults to ``0``.
        m (int): Number of edges added per node when constructing the
            Barabasi-Albert base graph (only relevant when
            ``base_graph`` is ``None``).  Defaults to ``9``.
        n (int, optional): Number of nodes for the Barabasi-Albert graph.
            Required when ``base_graph`` is ``None``.

    Returns:
        networkx.Graph: The thinned graph.

    Raises:
        AssertionError: If ``base_graph`` is ``None`` and ``n`` is also
            ``None``.
    """
    if(base_graph):
        graph = base_graph.copy()
    else:
        assert(n is not None), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = nx.barabasi_albert_graph(n=n, m=m)

# To get a graph with power-law-esque properties but without the fixed minimum degree,
# We modify the graph by probabilistically dropping some edges from each node.
    for node in graph:
        neighbors = list(graph[node].keys())
        quarantineEdgeNum = int(max(min(np.random.exponential(
            scale=scale, size=1), len(neighbors)), min_num_edges))
#        print(quarantineEdgeNum)
        quarantineKeepNeighbors = np.random.choice(
            neighbors, size=quarantineEdgeNum, replace=False)
        for neighbor in neighbors:
            if(neighbor not in quarantineKeepNeighbors):
                graph.remove_edge(node, neighbor)
    return graph


class RandomSingleGraphGenerator(GraphGenerator):
    """Graph generator producing a single-layer random contact graph.

    Builds one Barabasi-Albert graph (with exponential degree thinning) and
    assigns each edge a truncated-normal weight in ``(0, 1)``.  The resulting
    graph has no layer structure (it collapses all contacts into a single
    layer).

    Args:
        num_nodes (int): Number of nodes.  Defaults to ``10000``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`GraphGenerator`.
    """

    def __init__(self, num_nodes=10000, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes

        # generate random connections
        baseGraph = nx.barabasi_albert_graph(n=num_nodes, m=9)
        FG = custom_exponential_graph(baseGraph, scale=100)

#       list_of_zeroes = [ n for n, d in FG.degree() if d == 0 ]
#       if list_of_zeroes != []:
#           print('OMG: ', list_of_zeroes)

        # generate random weights from trunc norm
        lower, upper = 0, 1
        mu, sigma = 0.7, 0.3
        iii = 0
        for (u, v, d) in FG.edges(data=True):
            iii += 1
            FG[u][v]['weight'] = stats.truncnorm.rvs(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        print(iii, 'Edges')
        self.G = FG


class RandomGraphGenerator(GraphGenerator):
    """Graph generator producing independent random contact graphs per layer.

    For each of the 14 default layers a separate Barabasi-Albert graph is
    generated (with exponential degree thinning) and each edge is assigned a
    truncated-normal weight in ``(0, 1)`` with mean ``0.7`` and
    std-dev ``0.3``.  Layer-level transmission probabilities are also drawn
    from the same truncated-normal distribution.  All per-layer graphs are
    merged into the shared MultiGraph ``self.G``.

    The resulting graph has a mean degree of approximately 13 per layer.

    Args:
        num_nodes (int): Number of nodes in each per-layer graph.
            Defaults to ``10000``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`GraphGenerator`.
    """

    def __init__(self, num_nodes=10000, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes

        i = 0
        for l in self.layer_names:
            # generate random connections
            baseGraph = nx.barabasi_albert_graph(n=num_nodes, m=9)
            FG = custom_exponential_graph(baseGraph, scale=100)
            # generate random weights from trunc norm
            lower, upper = 0, 1
            mu, sigma = 0.7, 0.3
            for (u, v, d) in FG.edges(data=True):
                FG[u][v]['weight'] = stats.truncnorm.rvs(
                    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                FG[u][v]['type'] = l
            # other params of the graph
            FG.graph['layer_name'] = l
            self.G.graph['layer_probs'][i] = stats.truncnorm.rvs(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            FG.graph['layer_prob'] = self.G.graph['layer_probs'][i]
            self.Graphs[l] = FG
            i = i + 1
#            dot = nx.nx_pydot.to_pydot(FG)
#            print(dot)
        for l in self.layer_names:
            self.G.add_edges_from(self.Graphs[l].edges(data=True))

    def as_dict_of_graphs(self):
        """Return the pre-built per-layer graph dictionary.

        Overrides :meth:`GraphGenerator.as_dict_of_graphs` to return the
        dictionary built during ``__init__`` rather than rebuilding it.

        Returns:
            dict[str, networkx.Graph]: Mapping of layer name to its
            single-layer graph.
        """
        return self.Graphs


class PickleGraphGenerator(GraphGenerator):
    """Graph generator that restores a graph from a NetworkX pickle file.

    Args:
        path_to_pickle (str): Path to the pickle file containing a
            previously serialized NetworkX graph.  Defaults to
            ``'graph.pickle'``.
        **kwardgs: Additional keyword arguments (currently unused).
    """

    def __init__(self, path_to_pickle='graph.pickle', **kwardgs):
        self.read_pickle(path_to_pickle)


class CSVGraphGenerator(GraphGenerator):
    """Graph generator that builds a multi-layer graph from CSV files.

    Reads nodes, edges and layer definitions from three separate CSV files
    and populates the underlying NetworkX MultiGraph.  Duplicate edges (same
    ``type``, ``subtype``, endpoint pair and weight) are removed in debug
    mode.

    Class Attributes:
        layer_names (list): Starts empty; populated from the layers CSV.
        layer_probs (list): Starts empty; populated from the layers CSV.

    Args:
        path_to_nodes (str): Path to the nodes CSV file.  Defaults to
            ``'nodes.csv'``.
        path_to_edges (str): Path to the edges CSV file.  Defaults to
            ``'edges.csv'``.  Required columns: ``vertex1``, ``vertex2``,
            ``type``, ``subtype``, ``weight``.
        path_to_layers (str): Path to the layer-definition CSV file.
            Defaults to ``'etypes.csv'``.  Required columns: ``id``,
            ``name``, ``weight``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`GraphGenerator`.
    """

    layer_names = []
    layer_probs = []

    def __init__(self, path_to_nodes='nodes.csv', path_to_edges='edges.csv', path_to_layers='etypes.csv', **kwargs):
        super().__init__(**kwargs)

        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}
        nodes = pd.read_csv(path_to_nodes, **csv_hacking)
        edges = pd.read_csv(path_to_edges, **csv_hacking)
        layers = pd.read_csv(path_to_layers, **csv_hacking)

        indexNames = edges[edges['vertex1'] == edges['vertex2']].index
        if len(indexNames):
            print(f"Warning: dropping self edges!!!! {indexNames}")
            #            print(edges[edges['vertex1'] == edges['vertex2']])
            edges.drop(indexNames, inplace=True)

        #        print(layers)
        # fill the layers
#        layer_names = tuple(zip(layers.loc('id'), layers.loc('id2')))
        layers_to_add = layers.to_dict('list')
        self.layer_names = layers_to_add['name']
#        print(layers_names)
        self.layer_probs = layers_to_add['weight']
        self.layer_ids = layers_to_add['id']

        self.G.graph['layer_names'] = self.layer_names
        self.G.graph['layer_probs'] = self.layer_probs

        # fill the nodes
        nodes_to_add = nodes.to_dict('records')
        idx_s = list(range(0, len(nodes_to_add)))
        self.G.add_nodes_from(zip(idx_s, nodes_to_add))

        # fill the edges
        # get rid of duplicate edges
        if __debug__:
                    # fill the edges
            edges["e"] = edges.apply(
                lambda row: (
                    (int(row.vertex1), int(row.vertex2))
                    if int(row.vertex1) < int(row.vertex2)
                    else (int(row.vertex2), int(row.vertex1))
                ),
                axis=1
            )
            edges.drop_duplicates(inplace=True, subset=[
                                  "type", "subtype", "e", "weight"])
            edges.drop(columns=["e"], inplace=True)

        edges_to_add = edges.to_dict('list')
        froms = edges_to_add['vertex1']
        tos = edges_to_add['vertex2']
        datas = [{
            k: v
            for k, v in d.items()
            if k != 'vertex1' and k != 'vertex2'
        } for d in edges.to_dict('records')]
        self.G.add_edges_from(zip(froms, tos, datas))

    def __str__(self):
        """Return a human-readable string listing all edges with their data.

        Returns:
            str: Newline-separated representation of every edge in ``self.G``
            including edge attribute dictionaries.
        """
        return "\n".join([str(e) for e in self.G.edges(data=True)])
