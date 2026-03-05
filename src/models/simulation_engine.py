"""Plan-based discrete-time simulation engine.

This module defines :class:`SimulationEngine`, which replaces the Gillespie
propensity loop with a *planning* approach: each node holds a pre-scheduled
``(time_to_go, state_to_go)`` pair.  Every day the engine decrements counters
and moves nodes whose countdown has reached zero, then invokes
:meth:`daily_update` for state-dependent daily checks (e.g. infection
attempts).
"""

import numpy as np
import pandas as pd

import logging
import time
from models.engine import BaseEngine
from utils.history_utils import TimeSeries, TransitionHistory
import utils.global_configs as global_configs
from utils.global_configs import monitor


EXPECTED_NUM_DAYS = 300


class SimulationEngine(BaseEngine):
    """Discrete-time plan-based epidemic engine.

    Each simulated agent (node) carries a *plan*: a countdown
    (``time_to_go``) and a target state (``state_to_go``).  On every
    simulated day:

    1. :meth:`daily_update` is called for nodes that need a check
       (e.g. susceptible nodes that might be infected).
    2. All countdown timers are decremented.
    3. Nodes whose timer hits zero are moved to their planned state via
       :meth:`change_states`.
    4. :meth:`update_plan` sets new plans for the nodes that just moved.

    Subclasses implement :meth:`daily_update` and :meth:`update_plan` to
    define the model's disease-progression logic.

    Class-level attributes (override in subclasses):
        states (list): Ordered list of state codes.
        num_states (int): Number of states.
        state_str_dict (dict): State-code → label mapping.
        ext_code (int): State code used for external nodes.
        transitions (list): Allowed ``(from, to)`` pairs.
        num_transitions (int): Number of transitions.
        final_states (list): Absorbing states.
        invisible_states (list): States excluded from population count.
        unstable_states (list): States that can still change.
        fixed_model_parameters (dict): Scalar constructor parameters.
        model_parameters (dict): Per-node constructor parameters.
        common_arguments (dict): Common constructor parameters (seed, etc.).
    """

    states = []
    num_states = len(states)
    state_str_dict = {}

    ext_code = 0

    transitions = []
    num_transitions = len(transitions)

    final_states = []
    invisible_states = []
    unstable_states = []

    fixed_model_parameters = {}
    model_parameters = {}

    common_arguments = {
        "random_seed": (None, "random seed value"),
        "start_day": (1, "day to start")
    }

    def __init__(self, G, **kwargs):
        """Initialise the simulation engine on a contact graph.

        Args:
            G: Contact graph or multi-layer graph object.  Stored as both
                ``self.G`` (backward compatibility) and ``self.graph``.
            **kwargs: Keyword arguments that override any default declared in
                ``fixed_model_parameters``, ``model_parameters``, or
                ``common_arguments``.  State initial counts supplied as
                ``init_<STATE_LABEL>=<count>``.
        """
        self.G = G  # backward compatibility
        self.graph = G

        self.init_kwargs = kwargs

        # 2. model initialization
        self.inicialization()

        # 3. time and history setup
        self.setup_series_and_time_keeping()

        # 4. init states and their counts
        self.states_and_counts_init(ext_nodes=self.num_ext_nodes,
                                    ext_code=self.ext_code)


        # 5. set callback to None
        self.periodic_update_callback = None

        self.T = self.start_day - 1

    def update_graph(self, new_G):
        """Update the internal graph reference and derived node metadata.

        Safe to call with ``None`` (no-op).  Updates ``self.graph``,
        ``self.num_nodes``, ``self.num_ext_nodes``, and ``self.nodes``.

        Args:
            new_G: New graph object, or ``None`` to leave the graph unchanged.
        """
        if new_G is not None:
            self.G = new_G  # just for backward compability
            self.graph = new_G
            self.num_nodes = self.graph.num_nodes
            try:
                self.num_ext_nodes = self.graph.num_nodes - self.graph.num_base_nodes
            except AttributeError:
                #  for saved old graph
                self.num_ext_nodes = 0
            self.nodes = np.arange(self.graph.number_of_nodes).reshape(-1, 1)



    def inicialization(self):
        """Initialise model parameters and build node-index array.

        Delegates to the parent :meth:`inicialization`, then stores a
        ``(num_nodes, 1)`` array of node indices in ``self.nodes`` and
        caches ``self.num_nodes``.
        """
        super().inicialization()

        # node indexes
        self.nodes = np.arange(self.graph.num_nodes).reshape(-1, 1)
        self.num_nodes = self.graph.num_nodes

        

    def setup_series_and_time_keeping(self):
        """Create time-series buffers and per-node tracking arrays.

        Extends the parent setup with:

        * Event-log buffers (``tseries``, ``history``).
        * State-history array (size depends on ``global_configs.SAVE_NODES``).
        * Per-state duration lists (when ``global_configs.SAVE_DURATIONS``).
        * Per-node ``durations`` counter.
        * Per-state :class:`~utils.history_utils.TimeSeries` for counts and
          increments (pre-allocated to ``EXPECTED_NUM_DAYS`` entries).
        """
        super().setup_series_and_time_keeping()

        tseries_len = self.num_transitions * self.num_nodes

        self.tseries = TimeSeries(tseries_len, dtype=float)
        self.history = TransitionHistory(tseries_len)

        # state history
        if global_configs.SAVE_NODES:
            history_len = EXPECTED_NUM_DAYS
        else:
            history_len = 1
        self.states_history = TransitionHistory(
            history_len, width=self.num_nodes)

        if global_configs.SAVE_DURATIONS:
            self.states_durations = {
                s: []
                for s in self.states
            }

        self.durations = np.zeros(self.num_nodes, dtype=int)

        # state_counts ... numbers of inidividuals in given states
        self.state_counts = {
            state: TimeSeries(EXPECTED_NUM_DAYS, dtype=int)
            for state in self.states
        }

        self.state_increments = {
            state: TimeSeries(EXPECTED_NUM_DAYS, dtype=int)
            for state in self.states
        }

        # N ... actual number of individuals in population
        self.N = TimeSeries(EXPECTED_NUM_DAYS, dtype=float)


    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        """Initialise state counts and per-node planning arrays.

        Extends the parent :meth:`states_and_counts_init` with:

        * ``self.time_to_go`` – per-node countdown to next transition
          (``-1`` means "no scheduled transition").
        * ``self.state_to_go`` – planned next state for each node.
        * ``self.current_state`` – copy of the initial state assignment.
        * ``self.need_update`` – boolean flag per node indicating that the
          plan must be recomputed.

        Args:
            ext_nodes (int, optional): Number of external nodes. Defaults to
                ``None``.
            ext_code (int, optional): State code for external nodes. Defaults
                to ``None``.
        """
        super().states_and_counts_init(ext_nodes, ext_code)

        # time to go until I move to the state state_to_go
        self.time_to_go = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)
        self.state_to_go = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)

        self.current_state = self.states_history[0].copy().reshape(-1, 1)


        # need update = need to recalculate time to go and state_to_go
        self.need_update = np.ones(self.num_nodes, dtype=bool)

        
    def daily_update(self, nodes):
        """Perform daily per-node checks (e.g. infection attempts).

        Called once per day for nodes flagged in ``self.need_check``.
        No-op in the base class.  Subclasses override this to implement
        infection logic and other daily events.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes that require a
                daily check.
        """
        pass


    def change_states(self, nodes, target_state=None):
        """Move nodes to their planned (or a forced) target state.

        Clears the old membership, assigns the new state, updates
        ``state_counts``, ``state_increments``, and optionally
        ``states_history``, then calls :meth:`update_plan` so each node
        gets a fresh plan.

        Args:
            nodes (numpy.ndarray): Boolean bitmap indicating which nodes
                should change state.
            target_state (int, optional): If given, all *nodes* are moved
                to this state, ignoring ``self.state_to_go``.  If ``None``
                (default), each node is moved to its own ``self.state_to_go``
                value.
        """
        # discard current state
        self.memberships[:, nodes == True] = 0


        for node in nodes.nonzero()[0]:
            if target_state is None:
                new_state = self.state_to_go[node][0]
            else:
                new_state = target_state
            old_state = self.current_state[node, 0]

            self.memberships[new_state, node] = 1
            self.state_counts[new_state][self.t] += 1
            self.state_counts[old_state][self.t] -= 1
            self.state_increments[new_state][self.t] += 1
            if global_configs.SAVE_NODES:
                self.states_history[self.t][node] = new_state

        if target_state is None:
            self.current_state[nodes] = self.state_to_go[nodes]
        else:
            self.current_state[nodes] = target_state
        self.update_plan(nodes)

    def update_plan(self, nodes):
        """Generate new transition plans for nodes that just changed state.

        Sets ``self.time_to_go`` and ``self.state_to_go`` for each node in
        *nodes* based on the node's current state.  No-op in the base class.
        Subclasses override this to implement state-specific duration sampling.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes whose plans need
                updating.
        """
        pass

    def _get_target_nodes(self, nodes, state):
        """Return a bitmap of nodes that are both in *nodes* and in *state*.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of candidate nodes.
            state (int): State code to intersect with.

        Returns:
            numpy.ndarray: Boolean bitmap of shape ``(num_nodes,)`` that is
            ``True`` only where *nodes* is ``True`` and the node's current
            state equals *state*.
        """
        ret = nodes.copy().ravel()
        is_target_state = self.memberships[state, ret, 0]
        ret[nodes.flatten()] = is_target_state
        # ret = np.logical_and(
        #     self.memberships[state].flatten(),
        #     nodes.flatten()
        # )
        return ret
    
    def print(self, verbose=False):
        """Print the current calendar day and optionally per-state counts.

        Args:
            verbose (bool, optional): If ``True``, prints ``T`` (calendar day)
                and the count for every state.  Defaults to ``False``.
        """
        if verbose:
            print(f"T = {self.T} ({self.t})")
            for state in self.states:
                    print(f"\t {self.state_str_dict[state]} = {self.state_counts[state][self.t]}")

    def save_durations(self, f):
        """Write per-state duration lists to an open file as CSV rows.

        Args:
            f (file-like): Open writable file object.
        """
        for s in self.states:
            line = ",".join([str(x) for x in self.states_durations[s]])
            print(f"{self.state_str_dict[s]},{line}", file=f)

    def save_node_states(self, filename):
        """Save the per-node daily state history to a CSV file.

        If ``global_configs.SAVE_NODES`` is ``False``, logs a warning and
        returns an empty DataFrame.

        Args:
            filename (str): Destination file path.

        Returns:
            pandas.DataFrame: Empty DataFrame when node states were not saved.
        """
        if global_configs.SAVE_NODES is False:
            logging.warning(
                "Nodes states were not saved, returning empty data frame.")
            return pd.DataFrame()
        index = range(0, self.t+1)
        columns = self.states_history.values
        df = pd.DataFrame(columns, index=index)
        df.to_csv(filename)
        # df = df.replace(self.state_str_dict)
        # df.to_csv(filename)
        # print(df)

    def to_df(self):
        """Convert simulation output to a :class:`pandas.DataFrame`.

        Extends the parent :meth:`to_df` by adjusting the ``day`` column and
        the index when ``start_day`` is not 1.

        Returns:
            pandas.DataFrame: State-count and increment time-series with
            calendar-day index.
        """
        df = super().to_df()
        if self.start_day != 1:
            df["day"] = self.start_day + df["day"] - 1
            df.index = self.start_day + df.index - 1
        return df

    def run(self, T, print_interval=10, verbose=False):
        """Run the plan-based simulation for *T* days.

        Iterates over days 1..T, calling :meth:`run_iteration` each day and
        the periodic callback when set.  If the epidemic ends before *T* days,
        fills remaining days with the last observed counts.

        Args:
            T (int): Number of days to simulate.
            print_interval (int, optional): Print status every this many days;
                ``0`` or negative suppresses output. Defaults to ``10``.
            verbose (bool, optional): If ``True``, include per-state detail.
                Defaults to ``False``.

        Returns:
            bool: Always ``True``.
        """
        if global_configs.MONITOR_NODE is not None:
            monitor(0, f" being monitored, now in {self.state_str_dict[self.current_state[global_configs.MONITOR_NODE,0]]}")

        running = True
        self.tidx = 0
        self.T = self.start_day - 1
        if print_interval >= 0:
            self.print(verbose)

        for self.t in range(1, T+1):

            self.T = self.start_day + self.t - 1

            if __debug__ and print_interval >= 0 and verbose:
                print(flush=True)

            if (self.t >= len(self.state_counts[0])):
                # room has run out in the timeseries storage arrays; double the size of these arrays
                self.increase_data_series_length()

            if print_interval > 0 and verbose:
                start = time.time()
            running = self.run_iteration()

            # run periodical update
            if self.periodic_update_callback is not None:
                self.periodic_update_callback.run()

            if print_interval > 0 and (self.t % print_interval == 0):
                self.print(verbose)
                if verbose:
                    end = time.time()
                    print(f"Last day took: {end - start} seconds")

        if self.t < T:
            for t in range(self.t+1, T+1):
                if (t >= len(self.state_counts[0])):
                    self.increase_data_series_length()
                for state in self.states:
                    self.state_counts[state][t] = self.state_counts[state][t-1]
                    self.state_increments[state][t] = 0

        # finalize durations
        if global_configs.SAVE_DURATIONS:
            for s in self.states:
                durations = self.durations[self.memberships[s].flatten() == 1]
                durations = durations[durations != 0]
                self.states_durations[s].extend(list(durations))

        if print_interval >= 0:
            self.print(verbose)
        self.finalize_data_series()
        return True
 
    
    def run_iteration(self):
        """Perform one day of plan-based simulation.

        Steps performed each day:

        1. Copies previous-day state counts and resets increments.
        2. Increments all duration counters.
        3. Calls :meth:`daily_update` for nodes that need a check.
        4. Decrements ``self.time_to_go`` for all nodes.
        5. Moves nodes whose countdown reached zero via :meth:`change_states`.
        6. Saves duration statistics when configured.
        """
        logging.debug("DBG run iteration")

        # prepare
        # add timeseries members
        for state in self.states:
            self.state_counts[state][self.t] = self.state_counts[state][self.t-1]
            self.state_increments[state][self.t] = 0
        self.N[self.t] = self.N[self.t-1]

        self.durations += 1
        if global_configs.SAVE_NODES:
                self.states_history[self.t] = self.states_history[self.t-1]

        #print("DBG Time to go", self.time_to_go)
        #print("DBG State to go", self.state_to_go)

        # update times_to_go and states_to_go and
        # do daily_checkup
        self.daily_update(self.need_check)

        self.time_to_go -= 1
        #print("DBG Time to go", self.time_to_go)
        nodes_to_move = self.time_to_go == 0

        if global_configs.MONITOR_NODE and nodes_to_move[global_configs.MONITOR_NODE]:
            node = global_configs.MONITOR_NODE
            monitor(self.t,
                    f"changing state from {self.state_str_dict[self.current_state[node,0]]} to {self.state_str_dict[self.state_to_go[node,0]]}")

        orig_states = self.current_state[nodes_to_move]
        durs = self.durations[nodes_to_move.flatten()]
        self.change_states(nodes_to_move)
        self.durations[nodes_to_move.flatten()] = 0

        if global_configs.SAVE_DURATIONS:
            for s, d in zip(orig_states, durs):
                assert(d > 0)
                self.states_durations[s].append(d)



    
