"""Information-diffusion and rumour-spreading model classes.

This module defines several simulation models that reuse the
:class:`~models.simulation_engine.SimulationEngine` infrastructure for
information / opinion spreading rather than epidemic disease:

* :class:`STATES` / :class:`Tipping` – state-code enumerations.
* :class:`RumourModel` – basic SIR rumour-spreading model.
* :class:`RumourModelInfo` – extended rumour model with time-varying
  transmission and event-driven boost.
* :class:`InfoSIRModel` – SIR model driven by graph edge probabilities.
* :class:`InfoTippingModel` – threshold-based (tipping-point) adoption model.
"""

import json
import numpy as np
import pandas as pd

import time
import logging

from models.simulation_engine import SimulationEngine
from utils.random_utils import RandomDuration
from utils.random_utils import gen_tuple
from utils.history_utils import TimeSeries, TransitionHistory, ShortListSeries

from utils.global_configs import monitor
import utils.global_configs as global_configs


class STATES():
    """Simple SIR state codes for information-diffusion models.

    Attributes:
        S (int): Susceptible (uninformed).
        I (int): Infected / informed (spreading the rumour).
        R (int): Recovered (stopped spreading).
        EXT (int): External node.
    """
    S = 0
    I = 1 
    R = 2
    EXT = 10

class Tipping():
    """State codes for the threshold-based tipping-point model.

    Attributes:
        S (int): Susceptible (not yet adopted).
        ACTIVE (int): Active (adopted / tipped).
        EXT (int): External node.
    """

    S = 0
    ACTIVE = 1
    EXT = 10


class RumourModel(SimulationEngine):
    """Basic SIR rumour-spreading model on a contact network.

    Agents transition S → I when a neighbour is I (with rate ``lambda0``),
    and I → R after a fixed duration ``I_duration``.

    Inherits the plan-based stepping logic from
    :class:`~models.simulation_engine.SimulationEngine`.
    """
    states = [
        STATES.S,
        STATES.I,
        STATES.R,
        STATES.EXT
    ]

    num_states = len(states)
    state_str_dict = {
        STATES.S: "S",
        STATES.I: "I",
        STATES.R: "R",
        STATES.EXT: "EXT"
    }
    ext_code = STATES.EXT

    
    transitions = [
        (STATES.S, STATES.I),
        (STATES.I, STATES.R)
    ]

    num_transitions = len(transitions)

    final_states = [
        STATES.R
    ]

    invisible_states = [
        STATES.EXT
    ]

    unstable_states = [
        STATES.I
    ]

    fixed_model_parameters = {
        "I_duration": (1, "time in the I state"),
        "beta": (0,  "rate of transmission (exposure)")
    }

    def inicialization(self):
        """Delegate to parent initialiser (no extra setup needed)."""
        super().inicialization()

    def setup_series_and_time_keeping(self):
        """Delegate to parent time-series setup."""
        super().setup_series_and_time_keeping()

    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        """Initialise counts and flag S nodes for daily checks.

        Args:
            ext_nodes (int, optional): Number of external nodes.
            ext_code (int, optional): State code for external nodes.
        """
        super().states_and_counts_init(ext_nodes, ext_code)

        # need_check - state that needs regular checkup
        self.need_check = self.memberships[STATES.S]

        self.update_plan(np.ones(self.num_nodes, dtype=bool))

    def prob_of_contact(self, source_state, dest_state, beta):
        """Return nodes exposed by direct contact with I-state neighbours.

        Iterates over all undirected edges.  For each edge between a node in
        *source_state* and a node in *dest_state*, draws a Bernoulli coin with
        probability *beta*.  Returns the indices of source nodes that were
        exposed (may contain duplicates if a node has multiple I-neighbours).

        Args:
            source_state (int): State code of susceptible nodes.
            dest_state (int): State code of infectious nodes.
            beta (float): Per-edge transmission probability.

        Returns:
            numpy.ndarray: 1-D array of exposed node indices (possibly with
            duplicates).
        """

        # source_states - states that can be infected
        # dest_states - states that are infectious

        main_s = time.time()
        
        # is source in feasible state?
        is_relevant_source = self.memberships[source_state, self.graph.e_source, 0]
        
        # is dest in feasible state?
        is_relevant_dest = self.memberships[dest_state, self.graph.e_dest, 0]
        
        is_relevant_edge = np.logical_and(
            is_relevant_source,
            is_relevant_dest
        )

        assert type(beta) == float

        relevant_sources = self.graph.e_source[is_relevant_edge]
        relevant_dests = self.graph.e_dest[is_relevant_edge]

        # len(relevant_sources) == len(relevant_dests) --> for each edge that must be considered take source and dest
        # for each such edge draw a random number a test if < beta ==> get exposed nodes (the ones that are exposed at least once)

        num_relevant_edges = is_relevant_edge.sum() 
        r = np.random.rand(num_relevant_edges) 
        is_exposed = r < beta # for each relevant edge if transmission happens 

        is_exposed = is_exposed.ravel()
        
        exposed_nodes = relevant_sources[is_exposed]
        
        main_e = time.time()
        logging.info(f"PROBS OF CONTACT {main_e - main_s}")
        return exposed_nodes


    def daily_update(self, nodes):
        """Perform daily rumour-spreading check for susceptible nodes.

        Calls :meth:`prob_of_contact` and schedules exposed S nodes to move
        to I on the next day.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes needing a check.
        """
        # S
        target_nodes = self._get_target_nodes(nodes, STATES.S)

        # try infection
        exposed_nodes = self.prob_of_contact(
            STATES.S,
            STATES.I,
            self.lambda0,
        ).flatten()

        self.time_to_go[exposed_nodes] = 1
        self.state_to_go[exposed_nodes] = STATES.I

    def update_plan(self, nodes):
        """Set transition plans for nodes that just changed state.

        * S: no scheduled transition; flagged for daily checks.
        * I: scheduled to move to R after ``I_duration`` days.
        * R: no further transition.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes to update.
        """
        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, STATES.S)

        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = STATES.S
        self.need_check[target_nodes] = True

        # STATES.I:   "I"
        target_nodes = self._get_target_nodes(nodes, STATES.I)
        self.time_to_go[target_nodes] = self.I_duration
        self.state_to_go[target_nodes] = STATES.R
        self.need_check[target_nodes] = False

        # STATES.R:   "R",
        target_nodes = self._get_target_nodes(nodes, STATES.R)
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False

    def run_iteration(self):
        """Delegate to parent run_iteration (no extra logic needed)."""
        super().run_iteration()


class RumourModelInfo(RumourModel):
    """Extended rumour model with time-decaying transmission and event boost.

    Extends :class:`RumourModel` with:

    * A time-decaying per-node transmission rate:
      ``lambda = lambda0 * exp(-scale * time_in_I)``.
    * An optional event at time ``t_event`` that adds a temporary boost to
      lambda: ``lambda += event_boost * exp(-decay * (t - t_event))``.
    * A stochastic I → R transition each day (probability ``beta_duration``).

    The I state is maintained indefinitely until a daily Bernoulli trial
    (probability ``beta_duration``) triggers recovery.
    """

    fixed_model_parameters = {
        "lambda0": (0.001, "base rate of transmission"),
        "scale": (1.0, "scaling factor for the transmission probability"),
        "beta_duration": (0.02,"probability of ending the I state at each step"),
        #I_duration": (48, "time in the I state"),
        "t_event": (95, "time of the event that increases the spread"),
        "event_boost": (0.02, "boost in transmission rate after event"),
        "decay": (0.06, "decay rate of the increased spread after event"),
        "init_I": (12, "initial number of infected nodes")
    }
    
    def prob_of_contact(self, source_state, dest_state, beta):
        """Return newly exposed nodes using time-decaying lambda and event boost.

        Computes a per-node effective transmission rate ``lambda_`` based on:

        * How long each I node has been infectious (decay over time).
        * Whether the simulation has passed ``t_event`` (adds a boost).

        Then for each S node uses the compound probability formula
        ``1 - (1 - lambda_)^k`` where ``k`` is its count of I neighbours.

        Args:
            source_state (int): State code of susceptible nodes (S).
            dest_state (int): State code of infectious nodes (I).
            beta: Unused (included for API compatibility with parent).

        Returns:
            numpy.ndarray: 1-D array of exposed node indices.
        """
        # source_states - states that can be infected
        # dest_states - states that are infectious

        main_s = time.time()

        # is source in feasible state?
        is_relevant_source = self.memberships[source_state, self.graph.e_source, 0]

        # is dest in feasible state?
        is_relevant_dest = self.memberships[dest_state, self.graph.e_dest, 0]
        
        is_relevant_edge = np.logical_and(
            is_relevant_source,
            is_relevant_dest
        )

        #assert type(beta) == float

        # let's count lambda for all nodes, even not relevant ones 
        time_in_I = np.zeros(self.num_nodes)
        times = self.durations[self.memberships[STATES.I].ravel() == 1].ravel()
        time_in_I[self.memberships[STATES.I].ravel() == 1] = times
        lambda_ = self.lambda0 * np.exp(-self.scale * time_in_I)

            
        if self.t > self.t_event:
                lambda_ += self.event_boost * np.exp(-self.decay * (self.t - self.t_event))
                         
        relevant_sources = self.graph.e_source[is_relevant_edge]

        N = self.graph.number_of_nodes 
        counts = np.bincount(relevant_sources, minlength=N)

        #print(len(counts), self.memberships.shape[1], N)
        assert len(counts) == self.memberships.shape[1] == N 
        
        r = np.random.rand(N)
        # for all nodes, even those who are not relevant! for now  
        is_exposed = r <  1 - (1 - lambda_) ** counts

        exposed_nodes = np.arange(N)[is_exposed]

        main_e = time.time()
        logging.info(f"PROBS OF CONTACT {main_e - main_s}")
        return exposed_nodes

    def daily_update(self, nodes):
        """Perform daily spreading check and stochastic I → R transition.

        Calls the parent :meth:`RumourModel.daily_update`, then for each I
        node draws a Bernoulli coin (probability ``beta_duration``) to decide
        if it recovers today.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes needing a check.
        """
        super().daily_update(nodes)

        nodes_in_I = self._get_target_nodes(nodes, STATES.I)
        r = np.random.rand(len(nodes_in_I))
        end_I = r < self.beta_duration

        self.time_to_go[nodes_in_I] = -1
        self.time_to_go[end_I] = 1
        self.state_to_go[end_I] = STATES.R

    def update_plan(self, nodes):
        """Set transition plans for RumourModelInfo nodes.

        * S: no scheduled transition; flagged for daily checks.
        * I: no fixed deadline; flagged for daily checks (recovery is
          decided stochastically each day).
        * R: no further transition.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes to update.
        """
        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, STATES.S)

        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = STATES.S
        self.need_check[target_nodes] = True

        # STATES.I:   "I"
        target_nodes = self._get_target_nodes(nodes, STATES.I)
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = STATES.I
        self.need_check[target_nodes] = True

        # STATES.R:   "R",
        target_nodes = self._get_target_nodes(nodes, STATES.R)
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False


class InfoSIRModel(SimulationEngine):
    """SIR information-spreading model driven by graph edge probabilities.

    Uses the full edge-probability machinery from the graph object (rather than
    the simple per-edge coin flip of :class:`RumourModel`) to activate contacts
    each day.  The effective transmission rate per active edge is given by
    the per-node ``beta`` parameter multiplied by the edge intensity.

    Compartments: S (uninformed), I (spreading), R (stopped), EXT (external).
    """

    states = [
        STATES.S,
        STATES.I,
        STATES.R,
        STATES.EXT
    ]

    num_states = len(states)
    state_str_dict = {
        STATES.S: "S",
        STATES.I: "I",
        STATES.R: "R",
        STATES.EXT: "EXT"
    }
    ext_code = STATES.EXT

    
    transitions = [
        (STATES.S, STATES.I),
        (STATES.I, STATES.R)
    ]

    num_transitions = len(transitions)

    final_states = [
        STATES.R
    ]

    invisible_states = [
        STATES.EXT
    ]

    unstable_states = [
        STATES.I
    ]

    fixed_model_parameters = {
        "I_duration": (1, "time in the I state"),
    }

    model_parameters = {
                "beta": (0,  "rate of transmission (exposure)"),    
    }

    def inicialization(self):
        """Delegate to parent initialiser (no extra setup needed)."""
        super().inicialization()

    def setup_series_and_time_keeping(self):
        """Delegate to parent time-series setup."""
        super().setup_series_and_time_keeping()

    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        """Initialise counts and flag S nodes for daily infection checks.

        Args:
            ext_nodes (int, optional): Number of external nodes.
            ext_code (int, optional): State code for external nodes.
        """
        super().states_and_counts_init(ext_nodes, ext_code)

        # need_check - state that needs regular checkup
        self.need_check = self.memberships[STATES.S]

        self.update_plan(np.ones(self.num_nodes, dtype=bool))

    def prob_of_contact(self, source_state, dest_state, beta):
        """Compute per-node exposure probability using edge probabilities and intensities.

        Activates edges stochastically using the graph's edge-probability
        vector, then evaluates both directions for each active edge.  For
        edges connecting a *source_state* node to a *dest_state* node,
        applies a Bernoulli trial with rate ``beta[source] * intensity``.

        Args:
            source_state (int): State code of susceptible nodes.
            dest_state (int): State code of infectious nodes.
            beta (numpy.ndarray): Per-node transmission rate array.

        Returns:
            numpy.ndarray: Binary exposure vector of shape ``(num_nodes, 1)``;
            1 where a node was newly exposed.
        """

        # source_states - states that can be infected
        # dest_states - states that are infectious

        main_s = time.time()

        edges_probs = self.graph.get_all_edges_probs()
        num_edges = len(edges_probs)

        r = np.random.rand(num_edges)
        active_edges = (r < edges_probs).nonzero()[0]
        logging.info(f"active_edges {len(active_edges)}")

        source_nodes = self.graph.e_source[active_edges]
        dest_nodes = self.graph.e_dest[active_edges]
        types = self.graph.e_types[active_edges]
        # contact_info = (
        #     np.concatenate([source_nodes, dest_nodes]),
        #     np.concatenate([dest_nodes, source_nodes]),
        #     np.concatenate([types, types])
        # )
        # self.contact_history.append(contact_info)

        # take them in both directions
        n = len(active_edges)
        active_edges = np.concatenate([active_edges, active_edges])
        active_edges_dirs = np.ones(2*n, dtype=bool)
        active_edges_dirs[n:] = False

        source_nodes, dest_nodes = self.graph.get_edges_nodes(
            active_edges,
            active_edges_dirs
        )

        # is source in feasible state?
        is_relevant_source = self.memberships[source_state, source_nodes, 0]
        
        # is dest in feasible state?
        is_relevant_dest = self.memberships[dest_state, dest_nodes, 0]
        
        is_relevant_edge = np.logical_and(
            is_relevant_source,
            is_relevant_dest
        )

        ##########################
        relevant_edges = active_edges[is_relevant_edge]

        intensities = self.graph.get_edges_intensities(
            relevant_edges).reshape(-1, 1)
        relevant_sources, relevant_dests = self.graph.get_edges_nodes(
            relevant_edges, active_edges_dirs[is_relevant_edge])

        b_intensities = beta[relevant_sources]

        r = np.random.rand(intensities.ravel().shape[0]).reshape(-1, 1)
        is_exposed = r < (b_intensities * intensities)

        
        if np.all(is_exposed == False):
            return np.zeros((self.num_nodes, 1))

        is_exposed = is_exposed.ravel()
        
        exposed_nodes = relevant_sources[is_exposed]
        
        #print(exposed_nodes)
        
        ret = np.zeros((self.num_nodes, 1))
        ret[exposed_nodes] = 1

        main_e = time.time()
        logging.info(f"PROBS OF CONTACT {main_e - main_s}")
        return ret


    def daily_update(self, nodes):
        """Perform daily infection check for susceptible nodes.

        Calls :meth:`prob_of_contact` and schedules exposed S nodes to move
        to I on the next day.  External nodes are not yet supported.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes needing a check.

        Raises:
            NotImplementedError: If external nodes are present.
        """
        # S
        target_nodes = self._get_target_nodes(nodes, STATES.S)

        # if we have external nodes
        if self.num_ext_nodes > 0:
            raise NotImplementedError("External nodes not implemented yet.")

        # try infection
        P_infection = self.prob_of_contact(
                                      STATES.S,
                                      STATES.I,
                                      self.beta
                                      ).flatten()

        exposed = P_infection[target_nodes]

        exposed_mask = np.zeros(self.num_nodes, dtype=bool)
        exposed_mask[target_nodes] = (exposed == 1)

        print("Number of infected",  exposed_mask.sum())
        if exposed_mask.sum() > 0:
            print(exposed_mask)

        self.time_to_go[exposed_mask] = 1
        self.state_to_go[exposed_mask] = STATES.I

    def update_plan(self, nodes):
        """Set transition plans for InfoSIRModel nodes.

        * S: no scheduled transition; flagged for daily checks.
        * I: scheduled to move to R after ``I_duration`` days.
        * R: no further transition.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes to update.
        """
        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, STATES.S)

        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = STATES.S
        self.need_check[target_nodes] = True

        # STATES.I:   "I"
        target_nodes = self._get_target_nodes(nodes, STATES.I)
        self.time_to_go[target_nodes] = self.I_duration
        self.state_to_go[target_nodes] = STATES.R
        self.need_check[target_nodes] = False

        # STATES.R:   "R",
        target_nodes = self._get_target_nodes(nodes, STATES.R)
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False

    def run_iteration(self):
        """Delegate to parent run_iteration (no extra logic needed)."""
        super().run_iteration()

 
class InfoTippingModel(SimulationEngine):
    """Threshold-based (tipping-point) adoption model on a contact network.

    An agent adopts (S → ACTIVE) when the weighted fraction of its active
    neighbours exceeds its personal threshold ``theta``.  Once active, the
    agent remains active indefinitely.

    Edge weights are the graph's ``e_intensities`` values, and daily contact
    is stochastic (activated by edge probabilities).
    """

    states = [
        Tipping.S,
        Tipping.ACTIVE,
        Tipping.EXT
    ]

    num_states = len(states)

    state_str_dict = {
        Tipping.S : "S",
        Tipping.ACTIVE: "Active",
        Tipping.EXT: "Ext"
    }
    ext_code = STATES.EXT

    
    transitions = [
        (Tipping.S, Tipping.ACTIVE)
    ]

    num_transitions = len(transitions)

    model_parameters = {
                "theta": (0,  "threshold"),    
    }


    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        """Initialise counts and flag S nodes for daily tipping checks.

        Args:
            ext_nodes (int, optional): Number of external nodes.
            ext_code (int, optional): State code for external nodes.
        """
        super().states_and_counts_init(ext_nodes, ext_code)

        # need_check - state that needs regular checkup
        self.need_check = self.memberships[Tipping.S]

        self.update_plan(np.ones(self.num_nodes, dtype=bool))

    def _transmission(self):
        """Return a boolean bitmap of nodes that tip (S → ACTIVE) today.

        For each S node, activates edges stochastically, counts the weighted
        fraction of active neighbours, and flips the node to ACTIVE if the
        fraction exceeds ``self.theta[node]``.

        Returns:
            numpy.ndarray: Boolean array of shape ``(num_nodes,)``; ``True``
            where a node becomes active.
        """
        ret = np.zeros(self.num_nodes, dtype=bool)
        active_nodes = self.memberships[Tipping.S]

        edges_probs = self.graph.get_all_edges_probs()
        num_edges = len(edges_probs)
        print("num edges", num_edges)
        r = np.random.rand(num_edges)
        active_edges = (r < edges_probs) #.nonzero()[0] #bitmap
        
        for i, node in enumerate(self.graph.nodes):
            # bitmap of node's edges
            my_edges = self.graph.e_source == node
            print("my_edges", my_edges.shape)
            # keep only those that are active 
            my_edges = np.logical_and(
                my_edges,
                active_edges
            ).nonzero()[0]
            print("my_edges (after logical and)", my_edges.shape, my_edges.dtype)
            
            if len(my_edges) == 0:
                continue

            # take destination nodes
            my_neighbours = self.graph.e_dest[my_edges]
            print("my neighbours", my_neighbours.shape)
        
            active_neighbours = my_neighbours[(self.memberships[Tipping.ACTIVE] == 1)[my_neighbours].ravel()]
            print("active neighbours", active_neighbours.shape)

            my_edges_to_active = np.isin(self.graph.e_dest[my_edges], active_neighbours)

            print("my_edges_to_active", my_edges_to_active.shape)

            sum_all = self.graph.e_intensities[my_edges].sum()
            sum_active = self.graph.e_intensities[my_edges][my_edges_to_active].sum()

            if sum_active / sum_all > self.theta[node]:
                ret[i] = 1.0
        
        return ret 

    def daily_update(self, nodes):
        """Perform daily tipping check: activate nodes whose threshold is met.

        Calls :meth:`_transmission` and schedules newly tipping nodes to
        move to ACTIVE on the next day.  External nodes are not supported.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes needing a check.

        Raises:
            NotImplementedError: If external nodes are present.
        """
        # S
        target_nodes = self._get_target_nodes(nodes, Tipping.S)

        # if we have external nodes
        if self.num_ext_nodes > 0:
            raise NotImplementedError("External nodes not present in Tipping Model.")

        # try infection
        transmission = self._transmission().flatten()

        self.time_to_go[transmission] = 1
        self.state_to_go[transmission] = Tipping.ACTIVE

    def update_plan(self, nodes):
        """Set transition plans for InfoTippingModel nodes.

        * S: no scheduled transition; flagged for daily tipping checks.
        * ACTIVE: stays ACTIVE indefinitely; no further scheduled transition.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes to update.
        """
        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, Tipping.S)

        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = Tipping.S
        self.need_check[target_nodes] = True

        # STATES.Active:   "Active"
        target_nodes = self._get_target_nodes(nodes, Tipping.ACTIVE)
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = Tipping.ACTIVE
        self.need_check[target_nodes] = False



            
  
