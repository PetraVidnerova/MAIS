# NOTE: This is the older version used for TGMNetworkModel, not maintained now.

"""Legacy self-isolation policy for TGMNetworkModel (unmaintained).

This module provides the original :class:`WeeColdPolicy` that causes
symptomatic individuals to self-isolate with a configurable probability.
It is written for the older TGMNetworkModel and is **not maintained**.
Prefer :mod:`policies.wee_cold_sim` for current simulations.
"""

import numpy as np
import pandas as pd
from extended_network_model import STATES as states
from policy import Policy
from depo import Depo
from history_utils import TimeSeries
from quarantine_coefs import QUARANTINE_COEFS

import logging


class WeeColdPolicy(Policy):

    """Self-isolation policy for symptomatic individuals (legacy TGMNetworkModel).

    When a node transitions to the symptomatic state ``I_s`` during a
    day, it self-isolates with probability ``threshold``.  Isolated
    nodes are quarantined for ``duration`` days.  At the end of the
    quarantine period, nodes still symptomatic are kept for one
    additional day; asymptomatic nodes are released.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance (TGMNetworkModel).
    """

    def __init__(self, graph, model):
        """Initialise the self-isolation policy with default parameters.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
        """
        super().__init__(graph, model)

        # depo of quarantined nodes
        self.depo = Depo(graph.number_of_nodes)
        self.stopped = False
        self.threshold = 0.75
        self.duration = 7
        self.coefs = QUARANTINE_COEFS

    def to_df(self):
        """Return ``None`` (no statistics collected by this policy).

        Returns:
            None
        """
        return None

    def stop(self):
        """Signal the policy to stop quarantining new nodes.

        After calling ``stop``, ongoing quarantines are still managed
        but no new symptomatic nodes are quarantined.
        """
        self.stopped = True

    def quarantine_nodes(self, detected_nodes):
        """Place detected nodes into quarantine and suppress their graph layers.

        Args:
            detected_nodes (list): Node indices to quarantine.
        """
        if detected_nodes:
            assert self.coefs is not None
            self.graph.modify_layers_for_nodes(detected_nodes,
                                               self.coefs)
            self.depo.lock_up(detected_nodes, self.duration)

    def tick(self):
        """Advance the deposit by one day and return released nodes.

        Returns:
            numpy.ndarray: Indices of nodes released from quarantine today.
        """
        released = self.depo.tick_and_get_released()
        return released

    def release_nodes(self, released):
        """Restore graph edges for nodes leaving quarantine.

        Args:
            released (numpy.ndarray): Indices of nodes to release.
        """
        if len(released) > 0:
            logging.info(f"DBG {type(self).__name__} Released nodes: {released}")
            self.graph.recover_edges_for_nodes(released)

    def get_last_day(self):
        """Return the model-history slice for the current simulation day.

        Returns:
            list: Entries from ``model.history`` within the interval
            ``[current_day, current_day + 1)``.
        """
        current_day = int(self.model.t)
        start = np.searchsorted(
            self.model.tseries[:self.model.tidx+1], current_day, side="left")
        if start == 0:
            start = 1
        end = np.searchsorted(
            self.model.tseries[:self.model.tidx+1], current_day+1, side="left")
        return self.model.history[start:end]

    def run(self):
        """Execute one time-step of the self-isolation policy.

        Identifies nodes that became symptomatic today, applies the
        ``threshold`` coin-flip to decide who self-isolates, then
        advances the deposit and releases or extends quarantine for
        nodes whose isolation period has elapsed.
        """
        if self.stopped == True:
            return

        logging.info(
            f"Hello world! This is the wee cold function speaking.  {'(STOPPED)' if self.stopped else ''}")
        logging.info(f"QE {type(self).__name__}: Nodes in isolation/quarantine {self.depo.num_of_prisoners}")

        last_day = self.get_last_day()

        # those who became symptomatic today
        detected_nodes = [
            node
            for node, s, e in last_day
            if e == states.I_s and not self.depo.is_locked(node)
        ]
        logging.info(f"GDB Wee cold: got cold {len(detected_nodes)}")
        if len(detected_nodes) > 0:
            r = np.random.rand(len(detected_nodes))
            responsible = r < self.threshold
            detected_nodes = [
                d
                for i, d in enumerate(detected_nodes)
                if responsible[i]
            ]
        logging.info(f"GDB Wee cold: stay home {len(detected_nodes)}")

        # quarantine opens doors
        # newly detected are locked up
        released = self.tick()
        self.quarantine_nodes(detected_nodes)

        if len(released) > 0:
            still_symptomatic = released[self.is_I_s(released)]
            ready_to_leave = released[self.is_I_s(released) == False]

            if len(still_symptomatic) > 0:
                self.depo.lock_up(still_symptomatic, 1)
            if len(ready_to_leave) > 0:
                self.release_nodes(ready_to_leave)

    def is_I_s(self, node_ids):
        """Check whether nodes are currently in the symptomatic state I_s.

        Args:
            node_ids (numpy.ndarray): Indices of nodes to check.

        Returns:
            numpy.ndarray: Boolean array; ``True`` for nodes in state
            ``I_s``.
        """
        return self.model.memberships[states.I_s].ravel()[node_ids] == 1
