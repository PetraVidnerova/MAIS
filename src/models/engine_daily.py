"""Daily-batched Gillespie engine.

This module defines :class:`DailyEngine`, which runs the Gillespie event
selection intra-day but batches all state transitions to midnight, so the
observable state changes once per simulated day.
"""

import numpy as np
import scipy as scipy
import scipy.integrate
import networkx as nx
import time

from utils.history_utils import TimeSeries, TransitionHistory
from models.engine_seirspluslike import SeirsPlusLikeEngine


class DailyEngine(SeirsPlusLikeEngine):
    """Gillespie engine that applies state transitions only at midnight.

    Inherits from :class:`~models.engine_seirspluslike.SeirsPlusLikeEngine`.
    During the day the engine collects proposed transitions in a to-do list;
    at midnight (:meth:`update_states` / :meth:`midnight`) the transitions are
    committed in bulk, ensuring the observable model state changes only once
    per day.
    """

    def inicialization(self):
        """Initialise engine and allocate the daily to-do lists.

        Creates empty ``self.todo_list`` and ``self.todo_t`` accumulators
        before delegating to the parent initialiser.
        """
        self.todo_list = []
        self.todo_t = []

        super().inicialization()

    def run_iteration(self, alpha, cumsum, transition_types):
        """Sample the next Gillespie event and add it to the pending to-do list.

        Does *not* apply the transition immediately; it is deferred to the
        next call to :meth:`update_states`.  At most one pending transition
        per node is kept (first-event-wins within a day).

        Args:
            alpha (float): Total propensity (sum of all propensities).
            cumsum (numpy.ndarray): Cumulative-sum vector over flattened
                propensities, used for event selection.
            transition_types (list): Ordered list of ``(from_state,
                to_state)`` tuples matching the propensity order.

        Returns:
            bool: Always ``True`` (day-level termination is handled by
            :meth:`run`).
        """
        # 1. Generate 2 random numbers uniformly distributed in (0,1)
        r1 = np.random.rand()
        r2 = np.random.rand()

        # 2. Calculate propensities
        #        propensities, transition_types = self.calc_propensities()

        # Terminate when probability of all events is 0:
        # if propensities.sum() <= 0.0:
        #     self.finalize_data_series()
        #     return False

        # 4. Compute the time until the next event takes place
        tau = (1/alpha)*np.log(float(1/r1))
        self.t += tau

        # 5. Compute which event takes place
        transition_idx = np.searchsorted(cumsum, r2*alpha)
        transition_node = transition_idx % self.num_nodes
        transition_type = transition_types[int(transition_idx/self.num_nodes)]

        if transition_node not in [x[0] for x in self.todo_list]:
            #        if (transition_node, transition_type) not in self.todo_list:
            self.todo_t.append(self.t)
            self.todo_list.append((transition_node, transition_type))

        return True

    def update_states(self):
        """Commit all pending transitions accumulated during the current day.

        Iterates ``self.todo_list`` and applies each transition by updating
        ``self.memberships``, ``self.state_counts``, ``self.history``, and
        ``self.N``.  Clears the to-do lists afterwards.
        """
        #        print("updating states")
        # for t, (transition_node, transition_type) in zip(self.todo_t, self.todo_list):
        #     print(t, transition_node, "-->", transition_type)
        # 6. Update node states and data series
        for t, (transition_node, transition_type) in zip(self.todo_t, self.todo_list):
            self.tidx += 1
            if (self.tidx >= self.tseries.len()-1):
                # Room has run out in the timeseries storage arrays; double the size of these arrays
                self.increase_data_series_length()

            assert (self.memberships[transition_type[0], transition_node] == 1), (f"Assertion error: Node {transition_node} has unexpected current state, given the intended transition of {transition_type}.")

            self.memberships[transition_type[0], transition_node] = 0
            self.memberships[transition_type[1], transition_node] = 1
            self.tseries[self.tidx] = t
            self.history[self.tidx] = (transition_node, *transition_type)

            for state in self.states:
                self.state_counts[state][self.tidx] = self.state_counts[state][self.tidx-1]
            self.state_counts[transition_type[0]][self.tidx] -= 1
            self.state_counts[transition_type[1]][self.tidx] += 1

            self.N[self.tidx] = self.N[self.tidx-1]
            # if node died
            if transition_type[1] in (self.invisible_states):
                self.N[self.tidx] = self.N[self.tidx-1] - 1

        del self.todo_list
        del self.todo_t
        self.todo_list = []
        self.todo_t = []

    def midnight(self, verbose):
        """Execute end-of-day actions: commit transitions and recalculate propensities.

        Calls :meth:`update_states` to apply all pending transitions, fires
        ``self.periodic_update_callback`` if set (updating the graph if the
        callback returns a new one), then recomputes propensities for the next
        day via :meth:`propensities_recalc`.

        Args:
            verbose (bool): Passed through (currently unused).

        Returns:
            tuple: ``(alpha, cumsum, has_events, transition_types)`` as
            returned by :meth:`propensities_recalc`.
        """
        self.update_states()

        # run periodical update
        if self.periodic_update_callback:
            changes = self.periodic_update_callback(
                self.history, self.tseries[:self.tidx+1], self.t)

            if "graph" in changes:
                print("CHANGING GRAPH")
                self.update_graph(changes["graph"])

        return self.propensities_recalc()

    def print(self, verbose=False):
        """Print the current simulation time and optionally per-state counts.

        Args:
            verbose (bool, optional): If ``True``, also print per-state
                counts. Defaults to ``False``.
        """
        print("t = %.2f" % self.t)
        if verbose:
            for state in self.states:
                print(f"\t {self.state_str_dict[state]} = {self.current_state_count(state)}")
                print(flush=True)

    def propensities_recalc(self):
        """Recalculate propensities and return flattened cumulative-sum data.

        Returns:
            tuple: A 4-tuple ``(alpha, cumsum, has_events, transition_types)``
            where:

            * ``alpha`` (float) – total propensity.
            * ``cumsum`` (numpy.ndarray) – cumulative sum of flattened
              propensity array.
            * ``has_events`` (bool) – ``True`` if total propensity is > 0.
            * ``transition_types`` (list) – ordered ``(from, to)`` transition
              pairs.
        """
        # 2. Calculate propensities
        propensities = np.hstack(self.calc_propensities())
        transition_types = self.transitions

        # 3. Calculate alpha
        # nebylo by rychlejsi order=C a prohodi // a % ?
        propensities_flat = propensities.ravel(order="F")
        cumsum = propensities_flat.cumsum()
        alpha = propensities_flat.sum()
        return alpha, cumsum, propensities.sum() > 0.0, transition_types

    def run(self, T, print_interval=10, verbose=False):
        """Run the daily-batched simulation for up to *T* time units.

        Calls :meth:`propensities_recalc` once at the start, then loops over
        :meth:`run_iteration`.  At each midnight calls :meth:`midnight` to
        commit transitions and recompute propensities.

        Args:
            T (int or float): Duration to simulate.
            print_interval (int, optional): Print status every this many
                days. Defaults to ``10``.
            verbose (bool, optional): If ``True``, include per-state detail in
                progress messages. Defaults to ``False``.

        Returns:
            bool: ``True`` on completion, ``False`` if *T* <= 0.
        """
        if not T > 0:
            return False

        self.tmax += T

        running = True
        day = -1

        self.print(verbose=True)
        if print_interval > 0 and verbose:
            start = time.time()

        alpha, cumsum, running, transition_types = self.propensities_recalc()

        while running:

            running = self.run_iteration(alpha, cumsum, transition_types)

            # true after the first event after midnight
            day_changed = day != int(self.t)
            day = int(self.t)
            if day_changed and day != 0:
                alpha, cumsum, running, transition_types = self.midnight(
                    verbose)
                if print_interval > 0 and (day % print_interval == 0):
                    self.print(verbose)
                    if verbose:
                        end = time.time()
                        print("Last day took: ", end - start, "seconds")
                        start = time.time()

                # Terminate if tmax reached or num infectious and num exposed is 0:
                numI = sum([self.current_state_count(s)
                            for s in self.unstable_states
                            ])

                if self.t >= self.tmax or numI < 1:
                    self.finalize_data_series()
                    running = False

                day = int(self.t)

        self.print(verbose)
        self.finalize_data_series()
        return True

    # def increase_data_series_length(self):
    #     self.tseries.bloat()
    #     self.history.bloat()
    #     for state in self.states:
    #         self.state_counts[state].bloat()
    #     self.N.bloat()

    # def finalize_data_series(self):
    #     self.tseries.finalize(self.tidx)
    #     self.history.finalize(self.tidx)
    #     for state in self.states:
    #         self.state_counts[state].finalize(self.tidx)
    #     self.N.finalize(self.tidx)
