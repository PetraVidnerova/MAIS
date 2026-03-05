"""Base simulation engine for epidemic network models.

This module defines :class:`BaseEngine`, which provides the common
infrastructure shared by all concrete engines: graph management, parameter
broadcasting, time-series bookkeeping, and skeleton run/iteration hooks.
"""

import pandas as pd
import numpy as np
import scipy as scipy
import scipy.integrate
import networkx as nx
from pprint import pprint
from utils.history_utils import TimeSeries, TransitionHistory


class BaseEngine():
    """Abstract base class for all MAIS simulation engines.

    Subclasses must implement :meth:`run_iteration` (and usually
    :meth:`run`) to provide the concrete stepping logic.  This class
    provides:

    * Graph ingestion and adjacency-matrix construction
      (:meth:`update_graph`).
    * Model-parameter broadcasting to per-node arrays
      (:meth:`setup_model_params`).
    * Time-series and state-count initialisation
      (:meth:`setup_series_and_time_keeping`,
      :meth:`states_and_counts_init`).
    * Shared helper methods (:meth:`num_contacts`,
      :meth:`current_state_count`, :meth:`current_N`, etc.).
    * CSV export via :meth:`to_df` / :meth:`save`.
    """

    def setup_model_params(self, model_params_dict):
        """Broadcast scalar or list model parameters to per-node arrays.

        Each value in *model_params_dict* is stored as an attribute of shape
        ``(num_nodes, 1)``.  Scalars are broadcast; lists / arrays are
        reshaped.

        Args:
            model_params_dict (dict): Mapping of parameter name (str) to its
                value (scalar, list, or ``numpy.ndarray``).
        """
        # create arrays for model params
        for param_name, param in model_params_dict.items():
            if isinstance(param, (list, np.ndarray)):
                setattr(self, param_name,
                        np.array(param).reshape((self.num_nodes, 1)))
            else:
                setattr(self, param_name,
                        np.full(fill_value=param, shape=(self.num_nodes, 1)))

    def set_seed(self, random_seed):
        """Set the NumPy random seed for reproducible simulations.

        Args:
            random_seed (int): Seed value passed to ``numpy.random.seed``.
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed

    def inicialization(self):
        """Initialise model parameters and build the adjacency matrix.

        Reads values from ``self.init_kwargs`` (falling back to defaults
        declared in ``fixed_model_parameters``, ``common_arguments``, and
        ``model_parameters``), optionally seeds NumPy's random number
        generator, calls :meth:`update_graph` to build the adjacency matrix,
        and then calls :meth:`setup_model_params` to broadcast all model
        parameters to per-node arrays.
        """
        for argdict in (self.fixed_model_parameters,
                        self.common_arguments,
                        self.model_parameters):
            for name, definition in argdict.items():
                value = self.init_kwargs.get(name, definition[0])
                setattr(self, name, value)

        if self.random_seed:
            np.random.seed(self.random_seed)

        # setup adjacency matrix
        self.update_graph(self.G)

        model_params_dict = {
            param_name: self.__getattribute__(param_name)
            for param_name in self.model_parameters
        }
        self.setup_model_params(model_params_dict)

    def setup_series_and_time_keeping(self):
        """Initialise all time-tracking variables and empty time-series containers.

        Sets ``self.t``, ``self.tmax``, ``self.tidx`` to zero and creates
        ``None``-initialised containers for ``state_counts``,
        ``state_increments``, and ``self.N``.  Concrete subclasses override
        this method and fill the containers with proper
        :class:`~utils.history_utils.TimeSeries` objects.
        """
        self.t = 0
        tseries_len = 0
        self.expected_num_days = 30

        self.tseries = None
        self.meaneprobs = None
        self.medianeprobs = None

        self.history = None

        self.states_history = None
        self.states_durations = None

        # state_counts ... numbers of inidividuals in given states
        self.state_counts = None

        self.state_counts = {
            state: None
            for state in self.states
        }

        self.state_increments = {
            state: None
            for state in self.states
        }

        # N ... actual number of individuals in population
        self.N = TimeSeries(self.expected_num_days, dtype=float)

        # float time
        self.t = 0
        self.tmax = 0  # will be set when run() is called
        self.tidx = 0  # time index to time series

    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        """Initialise per-state node counts and the initial membership arrays.

        Reads ``init_<STATE_LABEL>`` entries from ``self.init_kwargs`` to set
        the starting counts for each state, assigns remaining nodes to the
        first state, shuffles node assignments randomly, and builds the
        ``self.memberships`` boolean array (shape: ``num_states × num_nodes ×
        1``).

        Args:
            ext_nodes (int, optional): Number of external nodes to pin to
                *ext_code* at the end of the node list. Defaults to ``None``.
            ext_code (int, optional): State code to assign to external nodes.
                Defaults to ``None``.

        Raises:
            ValueError: If external nodes are not the last nodes in the list.
        """

        self.init_state_counts = {
            s: self.init_kwargs.get(f"init_{self.state_str_dict[s]}", 0)
            for s in self.states
        }

        for state, init_value in self.init_state_counts.items():
            self.state_counts[state][0] = init_value

        for state in self.init_state_counts.keys():
            self.state_increments[state][0] = 0

        if ext_nodes is not None:
            self.state_counts[ext_code][0] = ext_nodes

        # add the rest of nodes to first state (S)
        nodes_left = self.num_nodes - sum(
            [self.state_counts[s][0] for s in self.states]
        )
        self.state_counts[self.states[0]][0] += nodes_left

        # invisible nodes does not count to population (death ones)
        self.N[0] = self.num_nodes - sum(
            [self.state_counts[s][0] for s in self.invisible_states]
        )

        # self.states_history[0] ... initial array of states
        start = 0
        for state, count in self.state_counts.items():
            self.states_history[0][start:start+count[0]].fill(state)
            start += count[0]
        # distribute the states randomly except ext nodes
        if ext_nodes is not None and ext_nodes != 0:
            if not np.all(self.states_history[0][-ext_nodes:] == ext_code):
                raise ValueError("External nodes should go last.")
            np.random.shuffle(self.states_history[0][:-ext_nodes])
        else:
            np.random.shuffle(self.states_history[0])

        # 0/1 num_states x num_nodes
        self.memberships = np.vstack(
            [self.states_history[0] == s
             for s in self.states]
        )
        self.memberships = np.expand_dims(self.memberships, axis=2).astype(bool)
        # print(self.memberships.shape)
        # print(np.all(self.memberships.sum(axis=0) == 1))
        # print(self.memberships.sum(axis=1))

        # print(self.states_history[0])
        # exit()

        self.durations = np.zeros(self.num_nodes, dtype="uint16")
        self.infect_start = np.zeros(self.num_nodes, dtype="uint16")
        self.infect_time = np.zeros(self.num_nodes, dtype="uint16")

    def update_graph(self, new_G):
        """Build the sparse adjacency matrix from the supplied graph object.

        Accepts a ``scipy.sparse.csr_matrix``, a ``numpy.ndarray``, or a
        ``networkx.Graph`` and stores the result as ``self.A`` (CSR format).
        Also updates ``self.num_nodes`` and ``self.degree``.

        Args:
            new_G: Contact network; one of ``scipy.sparse.csr_matrix``,
                ``numpy.ndarray``, or ``networkx.classes.graph.Graph``.

        Raises:
            TypeError: If *new_G* is not a supported type.
        """
        self.G = new_G

        if isinstance(new_G, scipy.sparse.csr_matrix):
            self.A = new_G
        elif isinstance(new_G, np.ndarray):
            self.A = scipy.sparse.csr_matrix(new_G)
        elif type(new_G) == nx.classes.graph.Graph:
            # adj_matrix gives scipy.sparse csr_matrix
            self.A = nx.adj_matrix(new_G)
        else:
            # print(type(new_G))
            raise TypeError(
                "Input an adjacency matrix or networkx object only.")

        self.num_nodes = self.A.shape[1]
        self.degree = np.asarray(self.node_degrees(self.A)).astype(float)

        # if TF_ENABLED:
        #     self.A = to_sparse_tensor(self.A)

    def node_degrees(self, Amat):
        """Return the degree (column sum) of each node in the adjacency matrix.

        Args:
            Amat (scipy.sparse.csr_matrix): Adjacency matrix of shape
                ``(num_nodes, num_nodes)``.

        Returns:
            numpy.ndarray: Array of shape ``(num_nodes, 1)`` with each node's
            degree.
        """
        # this is only for old types of model
        return Amat.sum(axis=0).reshape(self.num_nodes, 1)

    def set_periodic_update(self, callback):
        """Register a callback to be invoked at the end of each simulated day.

        The callback is stored as ``self.periodic_update_callback`` and is
        called by the engine's ``run`` loop at midnight of each simulated day.

        Args:
            callback (callable): Object or function to call once per day.
        """
        self.periodic_update_callback = callback
        #        print(f"DBD callback set {callback.graph}")

    def update_scenario_flags(self):
        """Recompute boolean scenario flags (testing, tracing, etc.).

        No-op in the base class.  Subclasses override this to update flags
        such as ``self.testing_scenario`` and ``self.tracing_scenario`` based
        on the current parameter values.
        """
        pass

    def num_contacts(self, state):
        """Return the number of contacts each node has in the given state(s).

        .. deprecated::
            Do not use this method in newer engines; it is retained only for
            backward compatibility with legacy model code.

        Args:
            state (str or list of str): State label(s) to count contacts in.
                A single string queries one state; a list sums over multiple
                states.

        Returns:
            numpy.ndarray: Column vector of shape ``(num_nodes, 1)`` with the
            contact count per node.

        Raises:
            TypeException: If *state* is neither a ``str`` nor a ``list``.
        """
        print("Warning: deprecated, do not use this method in newer engines.")

        if type(state) == str:
            # if TF_ENABLED:
            #     with tf.device('/GPU:' + "0"):
            #         x = tf.Variable(self.X == state, dtype="float32")
            #         return tf.sparse.sparse_dense_matmul(self.A, x)
            # else:
            return np.asarray(
                scipy.sparse.csr_matrix.dot(self.A, self.X == state))

        elif type(state) == list:
            state_flags = np.hstack(
                [np.array(self.X == s, dtype=int) for s in state]
            )
            # if TF_ENABLED:
            #     with tf.device('/GPU:' + "0"):
            #         x = tf.Variable(state_flags, dtype="float32")
            #         nums = tf.sparse.sparse_dense_matmul(self.A, x)
            # else:

            nums = scipy.sparse.csr_matrix.dot(self.A, state_flags)
            return np.sum(nums, axis=1).reshape((self.num_nodes, 1))
        else:
            raise TypeException(
                "num_contacts(state) accepts str or list of strings")

    def current_state_count(self, state):
        """Return the current count of nodes in *state*.

        Args:
            state (int): State code (see :class:`~models.states.STATES`).

        Returns:
            int or None: Current count, or ``None`` if the time-series has not
            been initialised yet.
        """
        if self.state_counts[state] is None:
            return None
        return self.state_counts[state][self.tidx]

    def current_N(self):
        """Return the current effective population size.

        Returns:
            float: Number of nodes not in any invisible state at the current
            time index.
        """
        return self.N[self.tidx]

    def increase_data_series_length(self):
        """Extend internal time-series storage when it is about to overflow.

        No-op in the base class.  Subclasses override this to call
        ``bloat()`` on their :class:`~utils.history_utils.TimeSeries`
        objects.
        """
        pass

    def finalize_data_series(self):
        """Trim all time-series to the actually used length after a run.

        No-op in the base class.  Subclasses override this to call
        ``finalize()`` on every :class:`~utils.history_utils.TimeSeries`.
        """
        pass

    def run_iteration(self):
        """Advance the simulation by one step.

        No-op in the base class.  Subclasses implement the concrete stepping
        logic here and return ``True`` while the simulation should continue or
        ``False`` when it should stop.

        Returns:
            bool: ``True`` to continue, ``False`` to stop.
        """
        pass

    def run(self, T, print_interval=10, verbose=False):
        """Run the simulation for *T* time units.

        No-op in the base class.  Subclasses implement the main simulation
        loop here.

        Args:
            T (int or float): Number of time units (days) to simulate.
            print_interval (int, optional): Print status every this many
                days. Defaults to ``10``.
            verbose (bool, optional): If ``True``, print per-state counts at
                each print interval. Defaults to ``False``.

        Returns:
            bool: ``True`` on successful completion.
        """
        pass

    def to_df(self):
        """Convert simulation output to a :class:`pandas.DataFrame`.

        Returns:
            pandas.DataFrame: Indexed by simulation time step (``T``), with
            one column per state (counts) and one column per state prefixed
            with ``inc_`` (increments), plus a ``day`` column.
        """
        index = range(0, self.t+1)
        col_increments = {
            "inc_" + self.state_str_dict[x]: col_inc.get_values()
            for x, col_inc in self.state_increments.items()
        }
        col_states = {
            self.state_str_dict[x]: count.get_values()
            for x, count in self.state_counts.items()
        }

        columns = {**col_states, **col_increments}
        #columns = col_states
        columns["day"] = np.floor(index).astype(int)
        # columns["mean_p_infection"] = self.meaneprobs.get_values()
        # columns["median_p_infection"] = self.medianeprobs.get_values()
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        return df

    def save(self, file_or_filename):
        """Save the simulation time-series to a CSV file.

        Calls :meth:`to_df` and writes the resulting DataFrame to disk.

        Args:
            file_or_filename (str or file-like): Destination path or open
                file object passed directly to
                :meth:`pandas.DataFrame.to_csv`.
        """
        df = self.to_df()
        df.to_csv(file_or_filename)
        print(df)

    def save_durations(self, file_or_filename):
        """Save per-state duration statistics to a file.

        Not yet implemented in :class:`BaseEngine`.  Subclasses may override
        this method to write duration histograms.

        Args:
            file_or_filename (str or file-like): Destination path or open
                file object.
        """
        print("Warning: self durations not implemented YET")

    def increase_data_series_length(self):  # noqa: F811 – second definition in original
        """Extend time-series storage (no-op; handled automatically by series objects).

        The underlying :class:`~utils.history_utils.TimeSeries` objects
        auto-extend, so this override is intentionally empty.
        """
        # this is done automaticaly by the series object now
        pass
        # for state in self.states:
        #     self.state_counts[state].bloat(100)
        #     self.state_increments[state].bloat(100)

        # self.N.bloat(100)
        # self.states_history.bloat(100)

    def increase_history_len(self):
        """Extend the transition-history buffers when storage is exhausted.

        No-op in the base class (handled automatically by the series objects
        in most engines).  Subclasses may override to call ``bloat()`` on
        ``self.tseries`` and ``self.history``.
        """
        # this is done automaticaly by the series object now
        pass
        # self.tseries.bloat(10*self.num_nodes)
        # self.history.bloat(10*self.num_nodes)

    def finalize_data_series(self):  # noqa: F811 – second definition in original
        """Trim all time-series to the actually consumed length (concrete implementation).

        Calls ``finalize`` on ``self.tseries``, ``self.history``, each
        per-state count and increment series, ``self.N``, and
        ``self.states_history``.
        """
        self.tseries.finalize(self.tidx)
        self.history.finalize(self.tidx)

        for state in self.states:
            self.state_counts[state].finalize(self.t)
            self.state_increments[state].finalize(self.t)
        self.N.finalize(self.t)
        self.states_history.finalize(self.t)

    def print(self, verbose=False):
        """Print the current simulation time and optionally per-state counts.

        Args:
            verbose (bool, optional): If ``True``, also print the current
                count for every state. Defaults to ``False``.
        """
        print("t = %.2f" % self.t)
        if verbose:
            for state in self.states:
                print(f"\t {self.state_str_dict[state]} = {self.current_state_count(state)}")
