"""Self-isolation policy for the current MAIS agent-based network model.

This module provides :class:`WeeColdPolicy`, which causes individuals
who develop symptoms (states ``I_s`` or ``S_s``) to self-isolate with
a configurable probability.  Unlike the legacy :mod:`policies.wee_cold`
module, this version works with the current
:mod:`models.agent_based_network_model` and supports loading parameters
from a configuration file.
"""

import numpy as np
import pandas as pd
from models.agent_based_network_model import STATES
from policies.policy import Policy
from policies.depo import Depo
from utils.history_utils import TimeSeries
from policies.quarantine_coefs import QUARANTINE_COEFS
from utils.config_utils import ConfigFile


import logging


class WeeColdPolicy(Policy):

    """Self-isolation policy for symptomatic individuals.

    When a node newly enters a symptomatic state (``I_s`` or ``S_s``),
    it self-isolates with probability ``threshold``.  The node's
    contact-network edges are suppressed for ``duration`` days.  At the
    end of the isolation period, nodes still symptomatic are kept for
    one additional day; asymptomatic nodes are released.

    Args:
        graph: The contact network graph object.  Quarantine
            coefficients are taken from ``graph.QUARANTINE_COEFS`` if
            available, otherwise the module-level defaults are used.
        model: The epidemic model instance.
        config_file (str, optional): Path to an INI-style configuration
            file.  The ``[SELFISOLATION]`` section may contain:

            * ``threshold`` – self-isolation probability in [0, 1]
              (default 0.5).
            * ``duration`` – isolation length in days (default 7).
    """

    def __init__(self, graph, model, config_file=None):
        """Initialise the self-isolation policy.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
            config_file (str, optional): Path to a configuration file.
        """
        super().__init__(graph, model)

        # depo of quarantined nodes
        self.depo = Depo(graph.number_of_nodes)
        self.stopped = False
        self.threshold = 0.5
        self.duration = 7

        if graph.QUARANTINE_COEFS is not None:
            self.coefs = graph.QUARANTINE_COEFS
        else:
            logging.warning("Using default quarantnine coefs.")
            self.coefs = QUARANTINE_COEFS

        self.symptomatic = np.zeros(model.num_nodes, dtype=bool).reshape(-1, 1)
        self.nodes = np.arange(model.num_nodes).reshape(-1, 1)

        # override defaults if config file provided
        if config_file:
            cf = ConfigFile()
            cf.load(config_file)
            my_config = cf.section_as_dict("SELFISOLATION")

            self.threshold = my_config.get("threshold", self.threshold)
            self.duration = my_config.get("duration", self.duration)

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
        if len(detected_nodes) > 0:
            assert self.coefs is not None
            self.graph.modify_layers_for_nodes(detected_nodes,
                                               self.coefs)
            self.depo.lock_up(detected_nodes, self.duration,
                              check_duplicate=True)

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
            logging.info(f"DBG {type(self).__name__} Released nodes: {len(released)}")
            self.graph.recover_edges_for_nodes(released)

    def run(self):
        """Execute one time-step of the self-isolation policy.

        Identifies nodes that are newly symptomatic (entered ``I_s`` or
        ``S_s`` for the first time today and not already in the deposit),
        applies the ``threshold`` coin-flip to decide who self-isolates,
        advances the deposit, and releases or extends isolation for nodes
        whose period has elapsed.
        """
        if self.stopped == True:
            return

        logging.info(
            f"Hello world! This is the wee cold function speaking.  {'(STOPPED)' if self.stopped else ''}")
        logging.info(f"QE {type(self).__name__}: Nodes in isolation/quarantine {self.depo.num_of_prisoners}")

        # those who became symptomatic today
        have_symptoms = np.logical_or(
            self.model.memberships[STATES.S_s],
            self.model.memberships[STATES.I_s]
        ).ravel()
        new_symptomatic = np.logical_and(
            have_symptoms,
            self.depo.depo == 0
        )

        # update symptomatic
#        print("symptomatic", self.symptomatic.shape)
#        print("new_symptomatic", new_symptomatic.shape)
        self.symptomatic[np.logical_not(have_symptoms)] = False
        self.symptomatic[new_symptomatic] = True

        detected_nodes = self.nodes[new_symptomatic].ravel()

#        print("detected nodes", detected_nodes)

        logging.info(f"GDB Wee cold: got cold {len(detected_nodes)}")
        if len(detected_nodes) > 0:
            r = np.random.rand(len(detected_nodes))
            responsible = r < self.threshold
            detected_nodes = detected_nodes[responsible.ravel()]
        logging.info(f"GDB Wee cold: stay home {len(detected_nodes)}")

        # quarantine opens doors
        # newly detected are locked up
        released = self.tick()
        self.quarantine_nodes(list(detected_nodes))

        if len(released) > 0:
            still_symptomatic = released[self.is_I_s(released)]
            ready_to_leave = released[self.is_I_s(released) == False]

            if len(still_symptomatic) > 0:
                self.depo.lock_up(still_symptomatic, 1)
            if len(ready_to_leave) > 0:
                self.release_nodes(ready_to_leave)

    def is_I_s(self, node_ids):
        """Check whether nodes are currently in a symptomatic state (I_s or S_s).

        Args:
            node_ids (numpy.ndarray): Indices of nodes to check.

        Returns:
            numpy.ndarray: Boolean array; ``True`` for nodes whose
            combined membership in states ``I_s`` and ``S_s`` equals 1.
        """
        return (self.model.memberships[STATES.I_s][node_ids] +
                self.model.memberships[STATES.S_s][node_ids]).ravel() == 1
