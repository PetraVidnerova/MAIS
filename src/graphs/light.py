"""Light-weight graph representation for agent-based network simulations.

This module provides the ``LightGraph`` class, which stores a multi-layer
undirected graph as a collection of NumPy arrays and a compressed sparse
row (CSR) adjacency matrix.  It is designed to be considerably faster than
NetworkX for the high-throughput edge queries required by epidemic and
information-spread models.

Typical workflow:

1. Build a ``LightGraph`` from CSV files with :meth:`LightGraph.read_csv`.
2. Pickle the result for repeated simulation runs.
3. Clone per-run state cheaply with :meth:`LightGraph.copy`.
"""

import logging
from operator import iconcat
from functools import reduce
from itertools import chain
import numexpr as ne
import numpy as np
import pandas as pd
from copy import copy
from scipy.sparse import csr_matrix, lil_matrix
import json

# import os
# os.environ["NUMEXPR_MAX_THREADS"] = "272"  # to fix numexpr complains


def concat_lists(l):
    """Return the concatenation of all lists contained in an iterable.

    Args:
        l (iterable): An iterable whose elements are lists (or other
            iterables) to be concatenated.

    Returns:
        list: A single flat list produced by chaining every element of ``l``.
    """
    return list(chain.from_iterable(l))


class LightGraph:
    """Graph for agent-based network models stored as NumPy arrays.

    NetworkX proved too slow for the high-throughput edge queries needed by
    large-scale simulations.  ``LightGraph`` therefore stores the graph as a
    collection of plain NumPy arrays together with a CSR adjacency matrix
    whose non-zero values are indices into those arrays.

    Typical usage is to load the graph from CSV files once with
    :meth:`read_csv` and then pickle the result for future simulation runs,
    because building from CSV is comparatively slow.

    Nodes carry integer IDs that are user-visible, but internally (and in the
    model) nodes are addressed by their positional index in the original
    nodes file.

    Attributes:
        random_seed (int or None): The seed passed to ``numpy.random.seed``
            at construction time, stored for reproducibility bookkeeping.
        edge_repo: Internal edge repository (populated by :meth:`read_csv`).
        A (scipy.sparse.csr_matrix or None): Adjacency matrix whose non-zero
            entries hold edge-repository keys.  ``None`` until
            :meth:`read_csv` has been called.
        is_quarantined (numpy.ndarray or None): Per-node quarantine counter.
            ``None`` until the first quarantine operation.

    Args:
        random_seed (int, optional): Seed for NumPy's random number
            generator.  ``None`` leaves the global state unchanged.
    """

    # __slots__ = ['e_types', 'e_subtypes', 'e_probs', 'e_intensities', 'e_source', 'e_dest', 'e_valid', 'edges_repo',
    # 'edges_directions', '__dict__'] # not really helpful here, beneficial only for lots of small objects
    def __init__(self, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        self.random_seed = random_seed
        self.edge_repo = None
        self.A = None
        self.is_quarantined = None

    def read_csv(self,
                 path_to_nodes='p.csv',
                 path_to_external=None,
                 path_to_layers=None,
                 path_to_edges='edges.csv',
                 path_to_quarantine=None,
                 path_to_layer_groups=None):
        """Load graph data from CSV files and build internal data structures.

        Reads node attributes, edge lists, layer definitions, optional
        external nodes, optional quarantine coefficients and optional layer
        groups.  After this call the graph is ready to use.

        Self-loops are silently dropped with a warning.

        Args:
            path_to_nodes (str): Path to the primary nodes CSV file.
                Defaults to ``'p.csv'``.
            path_to_external (str, optional): Path to an optional secondary
                nodes CSV file (e.g. external / virtual nodes).  When
                ``None`` no external nodes are added.
            path_to_layers (str, optional): Path to a CSV file defining
                layer IDs, names and weights.  When ``None`` a minimal
                two-layer default is used (layers 0 and 1).
            path_to_edges (str): Path to the edges CSV file.
                Defaults to ``'edges.csv'``.  Required columns are
                ``vertex1``, ``vertex2``; optional columns are ``layer``,
                ``sublayer``, ``probability`` and ``intensity`` (all
                back-filled with sensible defaults when absent).
            path_to_quarantine (str, optional): Path to a two-column
                headerless CSV file mapping layer ID to quarantine
                coefficient.  When ``None`` quarantine coefs are disabled.
            path_to_layer_groups (str, optional): Path to a JSON file
                defining named groups of layers.  When ``None`` layer groups
                are disabled.
        """

        csv_hacking = {'na_values': 'undef', 'skipinitialspace': True}
        base_nodes = pd.read_csv(
            path_to_nodes, **csv_hacking).drop_duplicates().reset_index()
        
        edges = pd.read_csv(path_to_edges, **csv_hacking)
        # ensure columns - for backward compatibility with model-M
        if "layer" not in edges.columns:
            edges["layer"] = 1
        if "sublayer" not in edges.columns:
            edges["sublayer"] = 0
        if "probability" not in edges.columns:
            edges["probability"] = 1.0
        if "intensity" not in edges.columns:
            edges["intensity"] = 1.0
        if path_to_layers is None:
            layers = pd.DataFrame([{"id": 0, "name": "no_layer", "weight": 1.0},
                                   {"id": 1, "name": "default", "weight": 1.0}])
        else:   
            layers = pd.read_csv(path_to_layers, **csv_hacking)
        if path_to_external is not None:
            external_nodes = pd.read_csv(path_to_external, **csv_hacking)
        else:
            external_nodes = pd.DataFrame()

        nodes = pd.concat([base_nodes, external_nodes],
                          ignore_index=True).drop(columns=["index"])

        if path_to_quarantine is None:
            self.QUARANTINE_COEFS = None
        else:
            df = pd.read_csv(path_to_quarantine, header=None, index_col=0)
            self.QUARANTINE_COEFS = {
                i: df.loc[i][1]
                for i in df.index
            }

        if path_to_layer_groups is not None:
            with open(path_to_layer_groups, "r") as f:
                self.LAYER_GROUPS = json.load(f)
        else:
            self.LAYER_GROUPS = None

        # layer names, ids and weights go to graph
        layers_to_add = layers.to_dict('list')

        self.layer_ids = layers_to_add['id']
        self.layer_name = layers_to_add['name']
        self.layer_weights = np.array(layers_to_add['weight'], dtype=float)

        # nodes
        # select categorical columns
        cat_columns = nodes.select_dtypes(['object']).columns
        nodes[cat_columns] = nodes[cat_columns].apply(
            lambda x: x.astype('category'))

        # save codes for backward conversion
        self.cat_table = {
            col: list(nodes[col].cat.categories)
            for col in cat_columns
        }

        # covert categorical to numbers
        nodes[cat_columns] = nodes[cat_columns].apply(
            lambda x: x.cat.codes)
        # pprint(nodes)

        # just test of conversion back
        # print(cat_columns)
        # for col in list(cat_columns):
        #     nodes[[col]] = nodes[[col]].apply(
        #         lambda x: pd.Categorical.from_codes(
        #             x, categories=cat_table[col])
        #     )
        # pprint(nodes)

        for col in nodes.columns:
            setattr(self, "nodes_"+col, np.array(nodes[col]))

        self.nodes = np.array(nodes.index)
        self.num_nodes = len(self.nodes)
        self.num_base_nodes = len(base_nodes)

        """ if self.num_nodes > 65535:
            raise ValueError(
                "Number of nodes too high (we are using unit16, change it to unit32 for higher numbers of nodes.")
 """
        #        self.ignored = set(external_nodes["id"])

        # edges
        # drop self edges
        indexNames = edges[edges['vertex1'] == edges['vertex2']].index
        if len(indexNames):
            logging.warning("Warning: dropping self edges!!!!")
            edges.drop(indexNames, inplace=True)

        # fill edges to a graph
        n_edges = len(edges)
        # edges data"
        self.e_types = np.empty(n_edges, dtype="uint32")
        self.e_subtypes = np.empty(n_edges, dtype="uint32")
        self.e_probs = np.empty(n_edges, dtype="float32")
        self.e_intensities = np.empty(n_edges, dtype="float32")
        self.e_source = np.empty(n_edges, dtype="uint32")
        self.e_dest = np.empty(n_edges, dtype="uint32")
        self.e_active = np.ones(n_edges, dtype="bool")

        # if value == 2 than is valid, other numbers prob in quarantine
        self.e_valid = 2 * np.ones(n_edges, dtype="float32")
        # edges repo which will eventually be list of sets and not a dict
        self.edges_repo = {
            0: []
        }
        self.edges_directions = {
            0: []
        }
        key = 1
        # working matrix
        tmpA = lil_matrix((self.num_nodes, self.num_nodes), dtype="uint32")

        forward_edge = True
        backward_edge = False

        id_dict = {self.nodes_id[i]: i for i in range(self.num_nodes)}

        # fill data and get indicies
        for i, row in enumerate(edges.itertuples()):
            self.e_types[i] = row.layer
            self.e_subtypes[i] = row.sublayer
            self.e_probs[i] = row.probability
            self.e_intensities[i] = row.intensity

            # if row.vertex1 in self.ignored or row.vertex2 in self.ignored:
            #     continue

            try:
                i_row = id_dict[row.vertex1]
                i_col = id_dict[row.vertex2]
            except IndexError:
                print("Node does not exist")
                print(row.vertex1, row.vertex2)
                print(np.where(self.nodes_id == row.vertex1),
                      np.where(self.nodes_id == row.vertex2))
                exit()

            i_row, i_col = min(i_row, i_col), max(i_row, i_col)

            self.e_source[i] = i_row
            self.e_dest[i] = i_col

            if tmpA[i_row, i_col] == 0:
                # first edge between (row, col)
                self.edges_repo[key], self.edges_directions[key] = [
                    i], forward_edge
                self.edges_repo[key + 1], self.edges_directions[key +
                                                                1] = [i], backward_edge
                tmpA[i_row, i_col] = key
                tmpA[i_col, i_row] = key + 1
                key += 2
            else:
                # add to existing edge list
                print("+", end="")
                key_forward = tmpA[i_row, i_col]
                key_backward = tmpA[i_col, i_row]
                self.edges_repo[key_forward].append(i)
                assert self.edges_directions[key_forward] == forward_edge
                # self.edges_directions[key_forward].append(forward_edge)
                self.edges_repo[key_backward].append(i)
                # self.edges_directions[key_backward].append(backward_edge)
                assert self.edges_directions[key_backward] == backward_edge

            if i % 10000 == 0:
                print("\nEdges loaded", i)

        # create matrix (A[i,j] is an index of edge (i,j) in array of edges)
        print("\nConverting lil_matrix A to csr ...", end="")
        self.A = csr_matrix(tmpA)
        print("level done")
        del tmpA

        print("Converting edges_repo to list ...", end="")
        # data = [None]
        # subedges_counts = [0]
        # for i_key in range(1, key):
        #     value_set = self.edges_repo[i_key]
        #     # if len(value_list) > 1:
        #     #     print(i_key)
        #     data.append(value_set)
        #     subedges_counts.append(len(value_set))
        # self.edges_repo = data
        # the above can be replaced by
        self.edges_repo = np.array(
            list(self.edges_repo.values()), dtype=object)
        subedges_counts = [len(s) for s in self.edges_repo]
        # subedges_counts = [len(s) for s in np.nditer(self.edges_repo, flags=['refs_ok'], op_flags=['readonly'])]
        print("level done")

        print("Converting edges_directions to list ... ", end="")
        data = [None]
        for i_key in range(1, key):
            dir_list = [self.edges_directions[i_key]] * subedges_counts[i_key]
            data.append(dir_list)
        self.edges_directions = np.array(data, dtype=object)
        print("level done")

        print("Control check ... ", end="")
        for i_key in range(1, key):
            assert len(self.edges_repo[i_key]) == len(
                self.edges_directions[i_key])
        print("ok")

        print("Precalculate array of layer weights ... ", end="")
        self.e_layer_weight = self.layer_weights[self.e_types]
        print("ok")
        print("LightGraph is ready to use.")

        logging.info(f"Max intensity {self.e_intensities.max()}")

    @property
    def number_of_nodes(self):
        """Total number of nodes in the graph (base + external).

        Returns:
            int: Number of nodes.
        """
        return self.num_nodes

    def get_nodes(self, layer):
        """Return all node indices that have at least one edge on the given layer.

        Args:
            layer (int): Layer ID to filter by.

        Returns:
            numpy.ndarray: Sorted array of unique node indices that participate
            in at least one edge whose layer equals ``layer``.
        """
        sources = self.e_source[self.e_types == layer]
        dests = self.e_dest[self.e_types == layer]
        return np.union1d(sources, dests)

    def get_edges_nodes(self, edges, edges_dirs):
        """Return the source and destination node indices for a set of edges.

        Edges are stored with an arbitrary orientation (smaller index as
        source).  ``edges_dirs`` corrects for this: when ``True`` the stored
        source is the logical source; when ``False`` the stored destination is
        the logical source.

        Args:
            edges (numpy.ndarray): 1-D array of edge indices.
            edges_dirs (numpy.ndarray): Boolean array of the same length as
                ``edges``.  ``True`` means *forward* direction (source →
                dest); ``False`` means *backward* direction.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A pair
            ``(source_nodes, dest_nodes)`` where each element is a 1-D
            integer array of node indices aligned with ``edges``.
        """
        sources = self.e_source[edges]
        dests = self.e_dest[edges]
        # sources, dests numpy vectors on nodes
        # edges_dirs - bool vector
        # if True take source if False take dest
        flags = edges_dirs
        # print(edges_dirs)
        source_nodes = sources * flags + dests * (1 - flags)
        dest_nodes = sources * (1 - flags) + dests * flags
        return source_nodes, dest_nodes

        #    def get_edges_subset(self, source_flags, dest_flags):
        #        active_subset = self.A[source_flags == 1, :][:, dest_flags == 1]
        #        edge_lists = [self.edges_repo[key] for key in active_subset.data]
        #        return subset, sum(edge_lists, [])

    def get_edges(self, source_flags, dest_flags, dirs=True):
        """Return all edges between two sets of nodes.

        Args:
            source_flags (numpy.ndarray): Binary vector of length
                ``num_nodes``.  A value of ``1`` marks nodes in the first
                (source) set.
            dest_flags (numpy.ndarray): Binary vector of length
                ``num_nodes``.  A value of ``1`` marks nodes in the second
                (destination) set.
            dirs (bool): When ``True`` (default) also return direction flags
                alongside the edge indices.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray] or numpy.ndarray:
            If ``dirs`` is ``True``, returns ``(edges, directions)`` where
            ``edges`` is a 1-D integer array of edge indices and
            ``directions`` is a corresponding boolean array (``True`` =
            forward).  If ``dirs`` is ``False``, only ``edges`` is returned.
            Both arrays are empty when no edges exist between the two sets.
        """
        active_subset = self.A[source_flags == 1, :][:, dest_flags == 1]
        active_edges_indices = active_subset.data
        if len(active_edges_indices) == 0:
            return np.array([]), np.array([])
        edge_lists = self.edges_repo[active_edges_indices]
        result = np.array(concat_lists(edge_lists))
        if dirs:
            dirs_lists = self.edges_directions[active_edges_indices]
            result_dirs = np.array(concat_lists(dirs_lists), dtype=bool)
            return result, result_dirs
        return result

    def get_nodes_edges(self, nodes):
        """Return all edge indices adjacent to a set of nodes.

        Args:
            nodes (array-like): Node indices whose incident edges are
                requested.

        Returns:
            list: Flat list of edge indices (possibly with duplicates when
            multiple nodes share an edge).  Returns an empty list when
            ``nodes`` is empty or when none of the nodes have any edges.
        """
        if len(nodes) == 0:
            return []
        active_subset = self.A[nodes]
        active_edges_indices = active_subset.data
        if len(active_edges_indices) == 0:
            logging.warning(f"Warning: no edges for nodes  {nodes}")
            return []
        edge_lists = self.edges_repo[active_edges_indices]
        result = concat_lists(edge_lists)
        return result

    def get_nodes_edges_on_layers(self, nodes, layers):
        """Return incident edge indices for a set of nodes filtered by layer.

        Identical to :meth:`get_nodes_edges` but restricts the result to
        edges whose layer ID appears in ``layers``.

        Args:
            nodes (array-like): Node indices whose incident edges are
                requested.
            layers (array-like): Allowed layer IDs.  Only edges on one of
                these layers are included in the result.

        Returns:
            list: Flat list of edge indices that belong to one of the
            specified layers.  Returns an empty list when no matching edges
            are found.
        """

        edges = self.get_nodes_edges(nodes)
        if len(edges) == 0:
            return edges
        edges_layers = self.e_types[edges]
        selected_edges = np.isin(edges_layers, layers)
        # todo: list or convert to numpy array?
        return [x
                for i, x in enumerate(edges)
                if selected_edges[i]
                ]

    def switch_off_edges(self, edges):
        """Permanently deactivate a list of edges.

        Sets ``e_active`` to ``False`` for each edge in ``edges``.  This
        operation is independent of quarantine state and sets the effective
        transmission probability to zero regardless of other factors.

        Args:
            edges (list): List of edge indices to deactivate.
        """
        assert type(edges) == list
        self.e_active[edges] = False

    def switch_on_edges(self, edges):
        """Reactivate a list of previously deactivated edges.

        Reverses :meth:`switch_off_edges` by setting ``e_active`` back to
        ``True``.  Does not interact with quarantine state.

        Args:
            edges (list): List of edge indices to reactivate.
        """
        assert type(edges) == list
        self.e_active[edges] = True

    def get_all_edges_probs(self):
        """Compute effective transmission probabilities for every edge.

        Combines the raw per-edge probability (``e_probs``), the quarantine
        validity flag (``e_valid``), the active flag (``e_active``) and the
        layer weight into a single scalar per edge.  Uses ``numexpr`` for
        fast vectorised evaluation.

        When ``e_valid == 2`` the edge is fully active and the probability
        is ``e_probs * layer_weight``.  Any other value of ``e_valid`` is
        interpreted as an override probability (e.g. from quarantine) and
        replaces ``e_probs`` directly.

        Returns:
            numpy.ndarray: Float array of length ``n_edges`` with effective
            transmission probabilities in ``[0, 1]``.
        """
        # probs = self.e_probs.copy()
        # invalid = self.e_valid != 2
        # probs[invalid] =  self.e_valid[invalid]
        # weights = self.layer_weights[self.e_types]
        # probs[self.e_active == False] = 0
        # #        return ne.evaluate("probs * weights")
        # return probs * weights
        return ne.evaluate("active * (e_probs * (e_valid == 2) + e_valid * (e_valid != 2)) * weights",
                           local_dict={
                               'active': self.e_active,
                               'e_probs': self.e_probs,
                               'e_valid': self.e_valid,
                               'weights': self.e_layer_weight
                           }
                           )

    def get_edges_probs(self, edges):
        """Compute effective transmission probabilities for a subset of edges.

        Applies the same logic as :meth:`get_all_edges_probs` but only for
        the requested edge indices.

        Args:
            edges (numpy.ndarray): 1-D integer array of edge indices.

        Returns:
            numpy.ndarray: Float array of the same length as ``edges`` with
            effective transmission probabilities in ``[0, 1]``.
        """
        assert type(edges) == np.ndarray
        layer_types = self.e_types[edges]
        probs = self.e_probs[edges] * (self.e_valid[edges] == 2)
        probs += self.e_valid[edges] * (self.e_valid[edges] != 2)
        weights = self.e_layer_weight[edges]
        return self.e_active[edges] * probs * weights

    def get_edges_intensities(self, edges):
        """Return the intensity values for a subset of edges.

        Args:
            edges (numpy.ndarray): 1-D integer array of edge indices.

        Returns:
            numpy.ndarray: Float array of the same length as ``edges``
            containing the raw intensity value (``e_intensities``) for each
            requested edge.
        """
        assert type(edges) == np.ndarray
        return self.e_intensities[edges]

    # these methods work only with hodonin's layers
    def is_super_edge(self, edges):
        """Identify super-spreader edges (Hodonin layer convention).

        Super-spreader edges are defined as those with layer ID >= 33.

        Args:
            edges (numpy.ndarray): 1-D integer array of edge indices.

        Returns:
            numpy.ndarray: Boolean array of the same length as ``edges``,
            ``True`` where the edge is a super-spreader edge.
        """
        assert type(edges) == np.ndarray
        etypes = self.e_types[edges]
        return etypes >= 33

    def is_family_edge(self, edges):
        """Identify household/family edges (Hodonin layer convention).

        Family edges are those with layer ID 1 or 2.

        Args:
            edges (numpy.ndarray): 1-D integer array of edge indices.

        Returns:
            numpy.ndarray: Boolean array of the same length as ``edges``,
            ``True`` where the edge belongs to a family layer.
        """
        assert type(edges) == np.ndarray
        etypes = self.e_types[edges]
        return np.logical_or(etypes == 1, etypes == 2)

    def is_class_edge(self, edges, all=True):
        """Identify school/class edges (Hodonin layer convention).

        School edges have layer IDs 4–11.  The ``all`` flag controls whether
        all school types are included or only lower school levels (layers
        4–7, excluding high school and higher elementary).

        Args:
            edges (numpy.ndarray): 1-D integer array of edge indices.
            all (bool): When ``True`` (default), include all school layers
                (4–11).  When ``False``, restrict to layers 4–7.

        Returns:
            numpy.ndarray: Boolean array of the same length as ``edges``,
            ``True`` where the edge is a class/school edge.
        """
        assert type(edges) == np.ndarray
        etypes = self.e_types[edges]
        if all:
            # all schools
            return np.logical_and(etypes >= 4, etypes <= 11)
        else:
            # excepty high and higher elementary
            return np.logical_and(etypes >= 4, etypes <= 7)

    def is_pub_edge(self, edges):
        """Identify pub/bar edges (Hodonin layer convention).

        Pub edges have layer IDs 20, 28 or 29.

        Args:
            edges (numpy.ndarray): 1-D integer array of edge indices.

        Returns:
            numpy.ndarray: Boolean array of the same length as ``edges``,
            ``True`` where the edge represents a pub/bar contact.
        """
        assert type(edges) == np.ndarray
        etypes = self.e_types[edges]
        # pubs
        return np.logical_or(etypes == 20,
                             np.logical_or(etypes == 28, etypes == 29))

    def modify_layers_for_nodes(self, node_id_list, what_by_what):
        """Apply quarantine by reducing edge probabilities for given nodes.

        For every edge incident to a node in ``node_id_list`` that is not
        already quarantined (``e_valid == 2``), the effective probability is
        replaced by ``e_probs * coef`` (clipped to ``[0, 1]``), where
        ``coef`` is the coefficient for that edge's layer taken from
        ``what_by_what``.

        The per-node quarantine counter ``is_quarantined`` is incremented so
        that edges shared with still-quarantined nodes are not accidentally
        restored by a partial release.

        Args:
            node_id_list (array-like): Positional indices of nodes to
                quarantine.
            what_by_what (dict): Mapping of ``{layer_id: coefficient}``
                specifying the multiplicative reduction per layer.  Must not
                be empty.

        Raises:
            ValueError: If ``what_by_what`` is falsy (empty or ``None``).
        """

        if self.is_quarantined is None:
            self.is_quarantined = np.zeros(self.number_of_nodes, dtype=int)

        self.is_quarantined[node_id_list] += 1

        if not what_by_what:
            raise ValueError("what_by_what missing or empty")

        relevant_edges = np.unique(self.get_nodes_edges(node_id_list))

        if len(relevant_edges) == 0:
            return

        # select edges to change (those who are not in quarantine)
        valid = self.e_valid[relevant_edges]
        relevant_edges = relevant_edges[valid == 2]
        edges_types = self.e_types[relevant_edges]

        logging.info(f"DBG edges that goes to quarantinexs {len(relevant_edges)}")
        # print(edges_types)
        for layer_type, coef in what_by_what.items():
            # print(layer_type)
            edges_on_this_layer = relevant_edges[edges_types == layer_type]
            # modify their probabilities
            self.e_valid[edges_on_this_layer] = self.e_probs[edges_on_this_layer] * coef
            # use out in clip to work inplace
            np.clip(self.e_valid[edges_on_this_layer], 0.0,
                    1.0, out=self.e_valid[edges_on_this_layer])

    def recover_edges_for_nodes(self, release):
        """Restore original edge probabilities when nodes leave quarantine.

        Decrements the quarantine counter for each node in ``release``.
        For nodes whose counter reaches zero (i.e. no longer quarantined on
        any count), edges that connect two fully released nodes have their
        ``e_valid`` flag reset to ``2`` (fully active).  Edges that still
        touch a quarantined node are left unchanged.

        Args:
            release (numpy.ndarray): Positional indices of nodes being
                released from quarantine.
        """

        self.is_quarantined[release] -= 1
        assert np.all(self.is_quarantined >= 0)
        no_quara = self.is_quarantined[release] == 0
        release = release[no_quara]

        if len(release) == 0:
            return
        relevant_edges = np.unique(self.get_nodes_edges(release))
        if len(relevant_edges) == 0:
            logging.warning("Warning: recovering nodes with no edges")
            return
        # from source and dest nodes select those who are not in quarantine
        source_nodes = self.e_source[relevant_edges]
        dest_nodes = self.e_dest[relevant_edges]

        is_quarrantined_source = self.is_quarantined[source_nodes]
        is_quarrantined_dest = self.is_quarantined[dest_nodes]

        # recover only edges where both nodes are free
        relevant_edges = relevant_edges[np.logical_not(
            np.logical_or(is_quarrantined_source, is_quarrantined_dest))]

        # recover probs
        self.e_valid[relevant_edges] = 2

    def final_adjacency_matrix(self):
        """Return self for backward compatibility with older graph interfaces.

        Earlier versions of the model expected a separate adjacency-matrix
        object.  This shim allows code written against that interface to work
        without modification.

        Returns:
            LightGraph: The graph instance itself.
        """
        return self

    def get_layer_for_edge(self, e):
        """Return the layer ID for a single edge.

        Args:
            e (int): Edge index.

        Returns:
            int: Layer ID of edge ``e``.
        """
        return self.e_types[e]

    def set_layer_weights(self, weights):
        """Replace the layer-weight vector and rebuild per-edge weight cache.

        The per-edge weight array ``e_layer_weight`` is recomputed in-place
        after the update so that subsequent probability calculations use the
        new weights immediately.

        Args:
            weights (array-like): New weight values indexed by layer ID.
                Must be the same length as the current ``layer_weights``
                array.
        """
        logging.info(f"DBG Updating layer weights {weights}")
        np.copyto(self.layer_weights, weights)
        #    print(self.layer_weights[i])
        self.e_layer_weight = self.layer_weights[self.e_types]

        logging.info(f"DBG new weigths {self.layer_weights}")

    def close_layers(self, list_of_layers, coefs=None):
        """Set the weight of named layers to zero (or a custom coefficient).

        Useful for simulating non-pharmaceutical interventions such as school
        or workplace closures.

        Args:
            list_of_layers (list[str]): Names of layers to close, matched
                against ``self.layer_name``.
            coefs (list[float], optional): Per-layer replacement weights.
                When provided, ``coefs[idx]`` is used instead of ``0`` for
                the layer at position ``idx`` in ``list_of_layers``.  When
                ``None`` all listed layers are set to weight ``0``.
        """
        print(f"Closing {list_of_layers}")
        for idx, name in enumerate(list_of_layers):
            i = self.layer_name.index(name)
            self.layer_weights[i] = 0 if not coefs else coefs[idx]
        print(self.layer_weights)

    def copy(self):
        """Return an optimised partial copy of the graph for a new simulation run.

        The vast majority of graph fields (topology, node attributes, edge
        metadata) are immutable between runs and are therefore shared via a
        shallow copy.  Only the mutable, run-specific arrays—``e_valid``,
        ``layer_weights``, ``e_active`` and ``e_layer_weight``—are deep-
        copied and reset to their default values.  The quarantine tracker
        ``is_quarantined`` is reset to ``None``.

        Returns:
            LightGraph: A new ``LightGraph`` instance that shares topology
            with the original but has independent mutable state.
        """
        heavy_fields = ['e_valid', 'layer_weights',
                        'e_active', 'e_layer_weight']
        new = copy(self)
        for key in heavy_fields:
            field = getattr(self, key)
            setattr(new, key, field.copy())
        new.is_quarantined = None
        new.e_valid.fill(2)
        new.e_active.fill(True)
        return new
