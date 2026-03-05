# NOTE: this is an older version of contact tracing policy for TGMNetworkModel,
# not maintained now!

"""Legacy contact-tracing policy for TGMNetworkModel (unmaintained).

This module provides older-style quarantine and contact-tracing policy
classes originally written for the TGMNetworkModel.  They are **not
maintained** and should not be used for new simulations.  Prefer
:mod:`policies.contact_tracing` for current work.

Classes:
    QuarantinePolicy: Base quarantine policy with contact tracing.
    EvaQuarantinePolicy: Full quarantine + enter test policy.
    CRLikeQuarantinePolicy: Czech-Republic-like variant.
    StrongEvaQuarantinePolicy: Maximum-riskiness variant.
    NoEvaQuarantinePolicy: Zero-riskiness variant.
    MiniEvaQuarantinePolicy: Minimal-riskiness variant.
    Exp2A/B/CQuarantinePolicy: Experimental riskiness variants.
    W10/20/30/40/60/80QuarantinePolicy: Uniform riskiness variants.
"""

import numpy as np
import pandas as pd
from extended_network_model import STATES as states
from policy import Policy
from depo import Depo
from history_utils import TimeSeries
from quarantine_coefs import QUARANTINE_COEFS, RISK_FOR_LAYERS, RISK_FOR_LAYERS_MAX, RISK_FOR_LAYERS_MINI, RISK_FOR_LAYERS_60, RISK_FOR_LAYERS_10, RISK_FOR_LAYERS_30
from quarantine_coefs import get_riskiness

#GIRL = 29691
GIRL = 10

DETECTED_STATES = (
    states.I_ds,
    states.E_d,
    states.I_da,
    states.I_dn,
    states.J_ds,
    states.J_dn
)


class QuarantinePolicy(Policy):

    """Base quarantine policy with optional contact tracing (legacy TGMNetworkModel).

    Provides the core deposit/release mechanism and contact-history
    filtering.  Intended as a base class; subclasses set ``coefs``,
    ``riskiness``, and ``duration`` before use.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance (TGMNetworkModel).
    """

    def __init__(self, graph, model):
        """Initialise the quarantine policy.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
        """
        super().__init__(graph, model)

        # depo of quarantined nodes
        self.depo = Depo(graph.number_of_nodes)

        self.coefs = None
        self.riskiness = None
        self.duration = None
        self.stopped = False
        self.days_back = 7

        self.stat_in_quarantine = TimeSeries(301, dtype=int)
        self.stat_detected = TimeSeries(301, dtype=int)
        self.stat_contacts_collected = TimeSeries(301, dtype=int)
        self.stat_released = TimeSeries(301, dtype=int)

        self.first_day = True
        # self.stat_in_quarantine[0] = 0
        # self.stat_detected[0] = 0
        # self.stat_contacts_collected[0] = 0
        # self.stat_released[0] = 0

    def to_df(self):
        """Return a DataFrame with quarantine and tracing statistics.

        Returns:
            pandas.DataFrame: DataFrame indexed by time ``T`` with
            columns for nodes in quarantine, detected nodes, contacts
            collected, and released nodes.
        """
        index = range(0, self.model.t+1)
        eva_name = type(self).__name__
        columns = {
            f"{eva_name}_nodes_in_quarantine":  self.stat_in_quarantine[:self.model.t+1],
            f"{eva_name}_detected_nodes": self.stat_detected[:self.model.t+1],
            f"{eva_name}_contacts_collected": self.stat_contacts_collected[:self.model.t+1],
            f"{eva_name}_released_nodes": self.stat_released[:self.model.t+1]
        }
        columns["day"] = np.floor(index).astype(int)
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        return df

    def stop(self):
        """Signal the policy to stop quarantining new nodes.

        After calling ``stop``, existing quarantines are still managed
        but no new nodes are placed into quarantine.
        """
        self.stopped = True

    def get_last_day(self):
        """Return the slice of model history corresponding to the current day.

        Returns:
            list: Entries from ``model.history`` that fall within the
            time interval ``[current_day, current_day + 1)``.
        """
        current_day = int(self.model.t)
        start = np.searchsorted(
            self.model.tseries[:self.model.tidx+1], current_day, side="left")
        if start == 0:
            start = 1
        end = np.searchsorted(
            self.model.tseries[:self.model.tidx+1], current_day+1, side="left")
        return self.model.history[start:end]

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
            print(f"DBG {type(self).__name__} Released nodes: {released}")
            self.graph.recover_edges_for_nodes(released)

    def do_testing(self, released):
        """Test the given nodes and classify them as healthy or ill.

        Args:
            released (numpy.ndarray): Indices of nodes to test.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A pair
            ``(healthy, still_ill)`` of node-index arrays based on
            whether each node is in a recovered/susceptible state.
        """

        print("DBG Leaving tests", len(released))
        if not len(released) > 0:
            return np.array([]), np.array([])

        # add number of tests to model statistics
        self.model.num_qtests[self.model.t] += len(released)

        # chose those that are not ill
        node_is_R = _is_R(released, self.model.memberships) == 1
        healthy = released[node_is_R]
        still_ill = released[node_is_R == False]

        # if GIRL in list(still_ill):
        #     print(
        #         f"ACTION LOG({int(self.model.t)}: node {29691} tested positivily.")

        return healthy,  still_ill

    def filter_contact_history(self, detected_nodes, days_back=None):
        """Filter contact history for detected nodes and apply riskiness weighting.

        Looks back ``days_back`` days in ``model.contact_history`` for
        edges where the source node is in ``detected_nodes``, then
        stochastically retains each contact with probability equal to
        the riskiness of its contact layer.

        Args:
            detected_nodes (list): Node indices whose contacts should be
                traced.
            days_back (int, optional): Number of days to look back.
                Defaults to ``self.days_back``.

        Returns:
            list: Contact node indices that passed the riskiness filter.
        """
        if days_back is None:
            days_back = self.days_back

        # if self.riskiness is None:
        #     return [
        #         contact[1]
        #         for contact_list in self.model.contact_history
        #         for contact in contact_list
        #         if contact[0] in detected_nodes
        #     ]
        # else:

        relevant_contacts = [
            (contact[1], _riskiness(contact[2], self.graph, self.riskiness))
            # five day back
            for contact_list in self.model.contact_history[-days_back:]

            for contact in contact_list
            if contact[0] in detected_nodes
        ]

        print(f"DBG QE {type(self).__name__}: all contacts {len(relevant_contacts)}")

        if not relevant_contacts:
            return relevant_contacts

        r = np.random.rand(len(relevant_contacts))
        selected_contacts = [
            contact
            for (contact, threashold), r_number in zip(relevant_contacts, r)
            if r_number < threashold
        ]
        print(f"DBG QE {type(self).__name__}: selected contacts {len(selected_contacts)}")
        return selected_contacts

    def select_contacts(self, detected_nodes, days_back=None):
        """Return the set of contacts for the given detected nodes.

        Args:
            detected_nodes (list): Node indices whose contacts should be
                traced.
            days_back (int, optional): Number of days to look back.
                Defaults to ``self.days_back``.

        Returns:
            set: Set of contact node indices.
        """
        return set(self.filter_contact_history(detected_nodes, days_back))


class EvaQuarantinePolicy(QuarantinePolicy):

    """Full quarantine policy with enter testing (legacy TGMNetworkModel).

    Detected nodes (those entering a detected state during the last
    day) are quarantined for ``duration`` days.  Their contacts are
    traced, placed in a phone-call waiting queue, then quarantined and
    given an enter test.  Contacts that test positive are moved to a
    detected state and trigger further contact tracing.  Leaving nodes
    are tested twice before release.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance (TGMNetworkModel).
    """

    def __init__(self, graph, model):
        """Initialise the Eva quarantine policy with default parameters.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
        """
        super().__init__(graph, model)
        self.enter_test = True
        self.coefs = QUARANTINE_COEFS
        self.duration = 14
        self.days_back = 7
        self.phone_call_delay = 2
        self.enter_test_range = [3, 4, 5]
        self.riskiness = get_riskiness(1.0, 0.8, 0.4)

        print(f"DBG {type(self).__name__}: Here we are.")
        print(f"DBG {type(self).__name__}: My riskiness: {self.riskiness}")

        # nodes marked as contacts and waiting for being informed
        self.waiting_room_phone_call = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test
        self.waiting_room_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test result
        self.waiting_room_result_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for second test
        self.waiting_room_second_test = Depo(graph.number_of_nodes)

    def run(self):
        """Execute one time-step of the Eva quarantine policy.

        On the first call, statistics arrays are back-filled with zeros.
        Each step: identifies newly detected nodes, traces their
        contacts, advances all waiting rooms, performs enter tests,
        quarantines all eligible nodes, runs exit tests, and releases
        passing nodes.
        """
        if self.first_day:
            self.stat_in_quarantine[0:self.model.t] = 0
            self.stat_detected[0:self.model.t] = 0
            self.stat_contacts_collected[0:self.model.t] = 0
            self.stat_released[0:self.model.t] = 0

            self.first_day = False

        print(
            f"Hello world! This is the eva policy function speaking.  {'(STOPPED)' if self.stopped else ''}")
        print(f"QE {type(self).__name__}: Nodes in isolation/quarantine {self.depo.num_of_prisoners}")

        if not self.stopped:
            last_day = self.get_last_day()

            # those who became infected today
            detected_nodes = [
                node
                for node, _, e in last_day
                if e in DETECTED_STATES and not self.depo.is_locked(node)
            ]
            self.stat_detected[self.model.t] = len(detected_nodes)
        else:
            detected_nodes = []

        # if GIRL in detected_nodes:
        #     print(
        #         f"ACTION LOG({int(self.model.t)}): node {GIRL} was detected and is quarantined by eva and asked for contacts.")

        new_contacts = self.select_contacts(detected_nodes)
        self.stat_contacts_collected[self.model.t] = len(new_contacts)

        print(f"{self.model.t} DBG QE: Qurantined nodes: {len(detected_nodes)}")
        print(f"{self.model.t} DBG QE: Found contacts: {len(new_contacts)}")

        # if GIRL in list(contacts):
        #     print(
        #         f"ACTION LOG({int(self.model.t)}): node {GIRL} was marked as contact.")

        # get contats to be quarantined (except those who are already in quarantine)
        contacts_ready_for_quarantine = [
            x
            for x in self.waiting_room_phone_call.tick_and_get_released()
            if not self.depo.is_locked(x)
        ]

        self.waiting_room_phone_call.lock_up(
            list(new_contacts), self.phone_call_delay, check_duplicate=True)

        print(
            f"{self.model.t} DBG QE: Quaratinted contacts: {len(contacts_ready_for_quarantine)}")

        # if GIRL in list(released_waiting_nodes):
        #     print(
        #         f"ACTION LOG({int(self.model.t)}): node {GIRL} was quarantined by Eva (because beeing contact).")

        # contacts wait for 5th day test and collect those who should be tested today
        if self.enter_test:
            nodes_to_be_tested = self.waiting_room_enter_test.tick_and_get_released()
            self.waiting_room_enter_test.lock_up(contacts_ready_for_quarantine,
                                                 np.random.choice(self.enter_test_range, size=len(
                                                     contacts_ready_for_quarantine))
                                                 )

            # do testing
            healthy, ill = self.do_testing(nodes_to_be_tested)
            # we do not care about healthy, ill waits for test results
            contacts_positively_tested = self.waiting_room_result_enter_test.tick_and_get_released()
            self.waiting_room_result_enter_test.lock_up(ill, 2)
            # move positivily tested to I/J_d* states
            for p in set(contacts_positively_tested):
                self.model.detected_node(p)
            # select contacts of positively tested
            other_contacts = self.select_contacts(
                contacts_positively_tested, self.days_back+2)
            self.stat_contacts_collected[self.model.t] += len(other_contacts)
            self.waiting_room_phone_call.lock_up(
                list(other_contacts), self.phone_call_delay, check_duplicate=True)

        # quarantine opens doors - those who spent the 14 days are released,
        # newly detected + contacted contacts are locked up
        released = self.tick()
        print(type(detected_nodes), detected_nodes)
        print(type(contacts_ready_for_quarantine),
              contacts_ready_for_quarantine)
        self.quarantine_nodes(
            detected_nodes+list(contacts_ready_for_quarantine))

        # do final testing
        # two positive tests are needed for leaving (tests are always correct!)
        second_test_candidates, still_ill = self.do_testing(released)

        # still ill nodes go back to quarantine
        if len(still_ill) > 0:
            for p in set(still_ill):
                self.model.detected_node(p)
            self.depo.lock_up(still_ill, 2)

        # if GIRL in list(prisoners):
        #     print(
        #         f"ACTION LOG({int(self.model.t)}): node {GIRL} waits for negative test in eva quarantine.")

        # healthy wait for the second test, those tested second time released
        ready_to_leave = self.waiting_room_second_test.tick_and_get_released()
        if len(second_test_candidates) > 0:
            self.waiting_room_second_test.lock_up(second_test_candidates, 2)
        self.model.num_qtests[self.model.t] += len(ready_to_leave)

        # if GIRL in list(release_candidates):
        #     print(
        #         f"ACTION LOG({int(self.model.t)}): node {GIRL} has negative test and waits for second one  in eva quarantine.")

        # if GIRL in list(really_released):
        #     print(
        #         f"ACTION LOG({int(self.model.t)}): node {GIRL} was released from quarantine by eva.")

        self.stat_released[self.model.t] = len(ready_to_leave)
        self.release_nodes(ready_to_leave)

        self.stat_in_quarantine[self.model.t] = self.depo.num_of_prisoners


class CRLikeQuarantinePolicy(EvaQuarantinePolicy):

    """Czech-Republic-like quarantine policy with time-varying riskiness.

    Applies harder riskiness early in the simulation and relaxes it at
    days 10 and 66.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        """Initialise with CR-like default riskiness.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
        """
        super().__init__(graph, model)
        self.riskiness = get_riskiness(1, 0.1, 0.01)
        self.enter_test = True

    def run(self):
        """Execute one time-step, adjusting riskiness on key days.

        Reconfigures riskiness at simulation days 10 and 66 before
        delegating to :meth:`EvaQuarantinePolicy.run`.
        """
        if self.model.t == 66:
            self.riskiness = get_riskiness(1.0, 0.8, 0.4)
            self.enter_test = True
        if self.model.t == 10:
            self.riskiness = get_riskiness(1.0, 0.6, 0.2)
        super().run()


class StrongEvaQuarantinePolicy(EvaQuarantinePolicy):

    """Eva quarantine policy with maximum riskiness for all layers.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = RISK_FOR_LAYERS_MAX


class NoEvaQuarantinePolicy(EvaQuarantinePolicy):

    """Eva quarantine policy with zero riskiness (no contacts traced).

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.0, 0.0, 0.0, 0.0)


class MiniEvaQuarantinePolicy(EvaQuarantinePolicy):

    """Eva quarantine policy with minimal riskiness (family only).

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = RISK_FOR_LAYERS_MINI


class Exp2AQuarantinePolicy(EvaQuarantinePolicy):

    """Experimental variant A: full riskiness for family, school/work, and leisure.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(1.0, 1.0, 1.0, 0.0)


class Exp2BQuarantinePolicy(EvaQuarantinePolicy):

    """Experimental variant B: full riskiness for family and school/work only.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(1.0, 1.0, 0.0, 0.0)


class Exp2CQuarantinePolicy(EvaQuarantinePolicy):

    """Experimental variant C: full riskiness for family contacts only.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(1.0, 0.0, 0.0, 0.0)


class W10QuarantinePolicy(EvaQuarantinePolicy):

    """Eva quarantine policy with uniform 10 % riskiness.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.1, 0.1, 0.1)


class W20QuarantinePolicy(EvaQuarantinePolicy):

    """Eva quarantine policy with uniform 20 % riskiness.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.2, 0.2, 0.2)


class W30QuarantinePolicy(EvaQuarantinePolicy):

    """Eva quarantine policy with uniform 30 % riskiness.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.3, 0.3, 0.3)


class W40QuarantinePolicy(EvaQuarantinePolicy):

    """Eva quarantine policy with uniform 40 % riskiness.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.4, 0.4, 0.4)


class W60QuarantinePolicy(EvaQuarantinePolicy):

    """Eva quarantine policy with uniform 60 % riskiness.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.6, 0.6, 0.6)


class W80QuarantinePolicy(EvaQuarantinePolicy):

    """Eva quarantine policy with uniform 80 % riskiness.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(0.8, 0.8, 0.8)


def _is_R(node_ids, memberships):
    """Check whether nodes are in a recovered or susceptible state.

    Args:
        node_ids (numpy.ndarray): Indices of nodes to check.
        memberships: Model membership array (state × nodes × 1).

    Returns:
        numpy.ndarray: Array of 0/1 values; 1 if the node is in a
        recovered/susceptible state (R_u, R_d, S, S_s, D_u, or D_d).

    Raises:
        AssertionError: If ``memberships`` is ``None``.
    """
    assert memberships is not None

    recovered_states_flags = (memberships[states.R_u] +
                              memberships[states.R_d] +
                              memberships[states.S] +
                              memberships[states.S_s]
                              + memberships[states.D_u] +
                              memberships[states.D_d])
    release_recovered = recovered_states_flags.ravel()[node_ids]
    return release_recovered


def _riskiness(contact, graph, riskiness):
    """Return the riskiness value for a given contact edge.

    Args:
        contact: Edge identifier used to look up the layer.
        graph: The contact network graph with a ``get_layer_for_edge``
            method.
        riskiness (numpy.ndarray): Per-layer riskiness array.

    Returns:
        float: Riskiness of the layer for the given contact edge.
    """
    #    print(f"DBG riskiness {graph.e_types[contact]}:{riskiness[graph.get_layer_for_edge(contact)]}")
    return riskiness[graph.get_layer_for_edge(contact)]
