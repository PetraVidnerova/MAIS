"""Contact-tracing policy for the MAIS epidemic simulation.

This module extends :class:`~policies.testing_policy.TestingPolicy`
with contact tracing.  Positively tested nodes are asked for their
recent contacts; those contacts receive a phone call, are placed into
quarantine, and undergo an enter test.  Contacts that test positive
trigger a second round of tracing.

Several pre-configured variants are provided (e.g.
:class:`CRLikePolicy`, :class:`StrongEvaQuarantinePolicy`, and a
family of ``W*QuarantinePolicy`` classes with uniform riskiness).
"""

import numpy as np
import pandas as pd
import logging
from itertools import chain

from policies.testing_policy import TestingPolicy
from utils.history_utils import TimeSeries
from models.agent_based_network_model import STATES


from policies.depo import Depo

from policies.quarantine_coefs import QUARANTINE_COEFS, RISK_FOR_LAYERS
from policies.quarantine_coefs import RISK_FOR_LAYERS_MAX, RISK_FOR_LAYERS_MINI, RISK_FOR_LAYERS_60, RISK_FOR_LAYERS_10, RISK_FOR_LAYERS_30
from policies.quarantine_coefs import get_riskiness

from utils.global_configs import monitor
import utils.global_configs as cfgs
from utils.config_utils import ConfigFile


class ContactTracingPolicy(TestingPolicy):

    """Testing policy extended with contact tracing.

    Individuals with symptoms go for a test with a certain probability.
    Those who test positive undergo contact tracing: their recent
    contacts are identified, placed in a phone-call waiting queue, then
    quarantined and given an enter test.  Contacts that test positive
    trigger a further round of contact tracing.

    Key configurable parameters (set directly or via config file):

    * ``duration`` – isolation length for confirmed cases (default 10).
    * ``duration_quara`` – quarantine length for contacts (default 14).
    * ``days_back`` – how many days of contact history to trace
      (default 7).
    * ``phone_call_delay`` – days between identification and quarantine
      notification (default 2).
    * ``enter_test_delay`` – days after last contact at which the enter
      test is performed (default 5).
    * ``riskiness`` – per-layer recall probability array.
    * ``auto_recover`` – skip exit test; release automatically.
    * ``enter_test`` – whether to perform an enter test for contacts.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
        config_file (str, optional): Path to an INI-style configuration
            file with sections ``[ISOLATION]``, ``[QUARANTINE]``, and
            ``[CONTACT_TRACING]``.
    """

    def __init__(self, graph, model, config_file=None):
        """Initialise the contact-tracing policy.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
            config_file (str, optional): Path to a configuration file.
        """
        super().__init__(graph, model)

        if graph.QUARANTINE_COEFS is not None:
            self.coefs = graph.QUARANTINE_COEFS
        else:
            self.coefs = QUARANTINE_COEFS

        self.riskiness = get_riskiness(graph, 1.0, 0.8, 0.4)
        self.duration = 10
        self.duration_quara = 14
        self.days_back = 7
        self.phone_call_delay = 2
        self.enter_test_delay = 5

        # if node already passed enter test and it was negative
        self.negative_enter_test = np.zeros(self.model.num_nodes, dtype=bool)
        # the day of the last contact with infectious node, -1 for undefined
        self.last_contact = np.full(
            self.model.num_nodes, fill_value=-1, dtype=int)

        self.auto_recover = False  # no final condititon
        self.enter_test = True    # do enter test

        if config_file is not None:
            self.load_config(config_file)

        logging.info(f"CONTACT TRACING POLICY: riskiness is {self.riskiness}")

        # nodes marked as contacts and waiting for being informed
        self.waiting_room_phone_call = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test
        self.waiting_room_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for enter-test result
        self.waiting_room_result_enter_test = Depo(graph.number_of_nodes)

        # nodes waiting for second test
        self.waiting_room_second_test = Depo(graph.number_of_nodes)

        # statistics
        self.stat_positive_enter_tests = TimeSeries(300, dtype=int)
        self.stat_contacts_collected = TimeSeries(300, dtype=int)

    def load_config(self, config_file):
        """Load policy parameters from a configuration file.

        Reads the ``[ISOLATION]``, ``[QUARANTINE]``, and
        ``[CONTACT_TRACING]`` sections and updates the corresponding
        instance attributes.

        Args:
            config_file (str): Path to the INI-style configuration file.
        """
        cf = ConfigFile()
        cf.load(config_file)

        isolation = cf.section_as_dict("ISOLATION")
        self.duration = isolation.get("duration", self.duration)

        quarantine = cf.section_as_dict("QUARANTINE")
        self.duration_quara = quarantine.get("DURATION", self.duration_quara)

        tracing = cf.section_as_dict("CONTACT_TRACING")
        cfg_riskiness = tracing.get("riskiness", None)
        if cfg_riskiness is not None:
            cfg_riskiness = list(map(float, cfg_riskiness))
            self.riskiness = get_riskiness(self.graph, *cfg_riskiness)
        self.days_back = tracing.get("days_back", self.days_back)
        self.phone_call_delay = tracing.get(
            "phone_call_delay", self.phone_call_delay)
        self.enter_test_delay = tracing.get(
            "enter_test_delay", self.enter_test_delay)
        self.auto_recover = tracing.get(
            "auto_recover", "Yes" if self.auto_recover else "No") == "Yes"
        self.enter_test = tracing.get(
            "enter_test", "Yes" if self.enter_test else "No") == "Yes"

    def first_day_setup(self):
        """Initialise statistics arrays and call the parent first-day setup.

        Fills ``stat_positive_enter_tests`` and
        ``stat_contacts_collected`` with zeros for all days before the
        policy starts, then delegates to
        :meth:`TestingPolicy.first_day_setup`.
        """
        # fill the days before start by zeros
        self.stat_positive_enter_tests[0:self.model.t] = 0
        self.stat_contacts_collected[0:self.model.t] = 0

        super().first_day_setup()

    def select_contacts(self, detected_nodes):
        """Trace recent contacts of detected nodes.

        For each detected node the contact history is searched back up
        to two days before first symptoms, or at most ``self.days_back``
        days.  Contacts are stochastically filtered by layer riskiness
        and external/dead nodes are excluded.  The ``last_contact``
        timestamp for each found contact is updated.

        Args:
            detected_nodes (numpy.ndarray): Indices of nodes whose
                contacts should be traced.

        Returns:
            set: Set of node indices identified as contacts.
        """
        contacts = set()

        if len(detected_nodes) == 0:
            return contacts

        # for each node trace back 2 days before first symtoms but
        # at most (or if there are no symptoms) self.days_back days
        # (no symptoms -> first_symptoms == -1)
        days_back_array = np.clip(
            (self.model.t - self.first_symptoms[detected_nodes]) + 2,
            0,
            self.days_back
        )

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in detected_nodes:
            monitor(self.model.t,
                    f"is beeing asked for contacts for {days_back_array[np.where(detected_nodes == cfgs.MONITOR_NODE)[0][0]]} days back.")

        for i, node in enumerate(detected_nodes):

            days_back = days_back_array[i]
            for t, day_list in enumerate(self.model.contact_history[-days_back:]):
                if day_list is None:  # first iterations
                    continue
                if len(day_list[0]) == 0:
                    continue
                # daylist is tuple (source_nodes, dest_nodes, types) including both directions
                my_edges = day_list[0] == node
                selected_contacts = day_list[1][my_edges]
                types = day_list[2][my_edges]

                selected_contacts = self.filter(selected_contacts, types)
                selected_contacts = self.filter_out_ext(selected_contacts)
                selected_contacts, _ = self.filter_dead(selected_contacts)

                self.last_contact[selected_contacts] = np.maximum(
                    self.last_contact[selected_contacts], self.model.t - days_back + t + 1)

                contacts.update(selected_contacts)

        return contacts

    def filter_out_ext(self, contacts):
        """Remove external (EXT-state) nodes from a contacts array.

        Args:
            contacts (numpy.ndarray): Array of node indices to filter.

        Returns:
            numpy.ndarray: Subset of ``contacts`` not in state ``EXT``.
        """
        is_not_ext = self.model.memberships[STATES.EXT, contacts, 0] == 0
        return contacts[is_not_ext]

    def filter(self, contacts, types):
        """Stochastically filter contacts by layer riskiness.

        Each contact is retained with probability equal to the
        riskiness of its contact-layer type (``self.riskiness[type]``).

        Args:
            contacts (numpy.ndarray): Array of node indices.
            types (numpy.ndarray): Integer array of layer types
                corresponding to each contact in ``contacts``.

        Returns:
            numpy.ndarray: Subset of ``contacts`` that are recalled.
        """
        risk = self.riskiness[types]
        r = np.random.rand(len(risk))
        return contacts[r < risk]

    def stop(self):
        """Signal the policy to stop quarantining new nodes.

        After calling ``stop``, existing quarantines are still managed
        but no new nodes are placed into isolation or quarantine.
        """
        self.stopped = True

    def select_test_candidates(self):
        """Select test candidates, excluding nodes already in isolation/quarantine.

        Delegates to the parent :meth:`TestingPolicy.select_test_candidates`
        and then filters out any node already held in the deposit.

        Returns:
            numpy.ndarray: Array of node indices to be tested today.
        """
        test_candidates = super().select_test_candidates()
        # exclude those that are already in quarantine
        return self.depo.filter_locked(test_candidates)

    def process_detected_nodes(self, target_nodes):
        """Isolate detected nodes, release finished quarantines, and run contact tracing.

        Workflow each time-step:

        1. Advance the deposit and process leaving nodes.
        2. Isolate ``target_nodes``.
        3. Trace contacts of ``target_nodes``; notify contacts via
           phone-call queue.
        4. Run enter-test procedure for contacts whose phone-call
           delay has elapsed.
        5. Trace contacts of enter-test positives and add them to the
           notification queue.

        Args:
            target_nodes (numpy.ndarray): Indices of nodes newly
                detected (positive test result) today.
        """

        released = self.tick()
        self.leaving_procedure(released)

        self.quarantine_nodes(target_nodes)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in target_nodes:
            monitor(self.model.t,
                    f"is sent to isolation.")

        # nodes asked for contacts
        contacts = self.select_contacts(target_nodes)

        # phone call
        contacted = self.waiting_room_phone_call.tick_and_get_released()
        if len(contacted) > 0:
            contacted = contacted[self.model.node_detected[contacted] == False]
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in contacted:
            monitor(self.model.t,
                    f"recieves a phone call and is sent to quarantine.")
        self.quarantine_nodes(contacted, last_contacts=True,
                              duration=self.duration_quara)

        # enter test
        positive_contacts = self.enter_test_procedure(contacted)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in positive_contacts:
            monitor(self.model.t,
                    f"has positive enter test.")

        # another contacts
        another_contacts = self.select_contacts(positive_contacts)

        # contacts contacted
        #all_contacts = np.union1d(contacts, another_contacts)
        #        all_contacts = np.array(list(set(chain(contacts, another_contacts))))
        all_contacts = np.array(list(contacts.union(another_contacts)))
        # if len(all_contacts) > 0:
        #     # filter out already detected
        #     all_contacts = all_contacts[self.model.node_detected[all_contacts] == False]
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in all_contacts:
            monitor(self.model.t,
                    f"is marked as contact and waits for phone call.")
        logging.debug(f"all contacts: {len(all_contacts)}")

        #all_contacts = np.array(list(set(contacts + another_contacts)))
        # print(contacts)
        # print(another_contacts)
        #all_contacts = np.union1d(contacts, another_contacts).astype(int)
        if len(all_contacts) > 0:
            all_contacts = self.depo.filter_locked(all_contacts)
            if len(all_contacts) > 0:
                # print(all_contacts)
                all_contacts = self.waiting_room_phone_call.filter_locked(
                    all_contacts)
                if len(all_contacts) > 0:
                    self.waiting_room_phone_call.lock_up(
                        all_contacts, self.phone_call_delay)
                    self.stat_contacts_collected[self.model.t] += len(
                        all_contacts)

    def enter_test_procedure(self, contacts):
        """Register contacts for an enter test and process those ready to be tested.

        ``contacts`` are added to the enter-test waiting room.  Nodes
        that have waited long enough are tested; positive results are
        counted and returned.

        Args:
            contacts (numpy.ndarray): Indices of newly quarantined
                contacts to register for an enter test.

        Returns:
            numpy.ndarray: Indices of contacts who tested positive
            during today's enter tests.
        """
        if not self.enter_test:
            return np.array([])

        to_be_tested = self.waiting_room_enter_test.tick_and_get_released()
        to_be_tested, _ = self.filter_dead(to_be_tested)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in to_be_tested:
            monitor(self.model.t,
                    f"goes for enter test.")

        contacts = self.waiting_room_enter_test.filter_locked(contacts)
        enter_test_delay = np.clip(
            (self.last_contact[contacts] - self.model.t) +
            self.enter_test_delay,
            1,
            self.enter_test_delay
        )

        self.waiting_room_enter_test.lock_up(contacts, enter_test_delay)
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in contacts:
            monitor(self.model.t,
                    f"is going to wait for enter test for {enter_test_delay[np.where(contacts == cfgs.MONITOR_NODE)[0][0]]}.")

        healthy, ill = self.perform_test(to_be_tested)
        if len(healthy) > 0:
            self.negative_enter_test[healthy] = True

        if len(ill) > 0:
            self.stat_positive_enter_tests[self.model.t] += len(ill)

        return ill

    def leaving_procedure(self, nodes):
        """Process nodes whose isolation/quarantine period has elapsed.

        If ``auto_recover`` is ``False``, nodes are tested:

        * Those still ill remain for two more days.
        * Those with a negative enter test are released immediately.
        * Those without a prior negative enter test wait two more days
          for a confirmatory second test.

        If ``auto_recover`` is ``True``, all nodes are released
        immediately without testing.

        Args:
            nodes (numpy.ndarray): Indices of nodes released from the
                deposit today (i.e. their timer reached zero).
        """

        if not self.auto_recover:
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes:
                monitor(self.model.t,
                        f"goes for leaving test.")

            nodes, dead = self.filter_dead(nodes)
            recovered, still_ill = self.perform_test(nodes)

            if len(still_ill) > 0:
                self.negative_enter_test[still_ill] = False
                self.depo.lock_up(still_ill, 2)

            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in still_ill:
                monitor(self.model.t,
                        f"will wait in quarantine/isolation another 2 days.")

            if len(recovered) > 0:
                to_release = recovered[self.negative_enter_test[recovered]]
                to_retest = recovered[self.negative_enter_test[recovered] == False]
            else:
                to_release = []
                to_retest = []

            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in to_release:
                monitor(self.model.t,
                        f"leaves qurantine/isolation with negative enter test and negative leaving test.")
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in to_retest:
                monitor(self.model.t,
                        f"waits for second leaving test.")

            released = self.waiting_room_second_test.tick_and_get_released()
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in released:
                monitor(self.model.t,
                        f"leaves quarantine/isolation with two negative leaving tests.")

            if len(to_retest) > 0:
                self.waiting_room_second_test.lock_up(to_retest, 2)

            if len(to_release) > 0:
                released = np.union1d(released, to_release)

        else:
            # auto recover
            released = nodes
            dead = None
            self.node_active_case[released] = False

        if len(released) > 0:
            logging.info(f"releasing {len(released)} nodes from isolation")
            self.release_nodes(released)
            self.negative_enter_test[released] = False  # for next time

        if dead is not None and len(dead) > 0:
            logging.info(f"releasing {len(released)} dead nodes from isolation")
            self.release_nodes(dead)
            self.node_active_case[dead] = False

    def run(self):
        """Execute one time-step of the contact-tracing policy.

        Resets daily statistics counters and then delegates to the
        parent :meth:`TestingPolicy.run`.
        """
        self.stat_positive_enter_tests[self.model.t] = 0
        self.stat_contacts_collected[self.model.t] = 0
        super().run()

    def to_df(self):
        """Return a DataFrame with testing and contact-tracing statistics.

        Extends the parent :meth:`TestingPolicy.to_df` result with
        columns ``positive_enter_test`` and ``contacts_collected``.

        Returns:
            pandas.DataFrame: DataFrame indexed by time ``T``.
        """
        df = super().to_df()

        df["positive_enter_test"] = self.stat_positive_enter_tests[:self.model.t+1]
        df["contacts_collected"] = self.stat_contacts_collected[:self.model.t+1]

        return df


# variants - should be done using config

class CRLikePolicy(ContactTracingPolicy):

    """Contact-tracing policy that mimics the Czech Republic (CR) approach.

    Applies a time-varying riskiness schedule that adjusts contact-recall
    probabilities at several hard-coded simulation days to reflect
    changing intervention intensity.  If a config file is provided it
    overrides the built-in schedule.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
        config_file (str, optional): Path to a configuration file that
            overrides the built-in riskiness schedule.
    """

    def __init__(self, graph, model, config_file=None):
        """Initialise the CR-like policy with a time-varying riskiness schedule.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
            config_file (str, optional): Path to a configuration file.
        """
        super().__init__(graph, model, config_file=None)  # run it without loading config
        self.riskiness = get_riskiness(graph, 1, 0.1, 0.01)
        self.enter_test = True

        self.auto_recover = False
        # reconfig
        for date in 10, 66, 125, 188, 300, 311:
            if self.model.T >= date:
                self.reconfig(date)

        # if config file provided, override the setup
        # do not forget about reconfig
        if config_file is not None:
            self.load_config(config_file)

    def reconfig(self, day):
        """Apply riskiness configuration for a specific simulation day.

        Called both during ``__init__`` (to replay past reconfiguration
        events when the simulation starts mid-run) and during ``run``
        (once per day).

        Args:
            day (int): Simulation day number at which to apply changes.
        """
        if day == 66:
            self.riskiness = get_riskiness(self.graph, 1.0, 0.8, 0.4)
            self.enter_test = True

        if day == 10:
            self.riskiness = get_riskiness(self.graph, 1.0, 0.6, 0.2)

        if day == 125:
            self.riskiness = get_riskiness(self.graph, 0.8, 0.6, 0.2)
            self.auto_recover = True

        if day == 188:
            #            self.riskiness = get_riskiness(0.6, 0.4, 0.1)
            self.riskiness = get_riskiness(self.graph, 0.5, 0.1, 0.05)
           # self.riskiness = get_riskiness(0, 0, 0)
            self.auto_recover = True

        # if day == 300:
        #     self.riskiness = get_riskiness(0.0, 0.0, 0.0)
        #     self.auto_recover = True

        # if day == 311:
        #     self.riskiness = get_riskiness(0.0, 0.0, 0.0)
        #     self.auto_recover = True

    def run(self):
        """Execute one time-step, applying any scheduled reconfiguration first.

        Calls :meth:`reconfig` for the current simulation day and then
        delegates to the parent :meth:`ContactTracingPolicy.run`.
        """
        # if self.model.T == 218:
        #     self.riskiness = get_riskiness(0.8, 0.4, 0.2)
        self.reconfig(self.model.T)
        super().run()


# do not use these classes (abandoned), use config file
class StrongEvaQuarantinePolicy(ContactTracingPolicy):

    """Contact-tracing policy with maximum riskiness for all layers.

    Abandoned variant; prefer using a config file instead.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = RISK_FOR_LAYERS_MAX


class NoEvaQuarantinePolicy(ContactTracingPolicy):

    """Contact-tracing policy with zero riskiness (no contacts traced).

    Abandoned variant; prefer using a config file instead.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.0, 0.0, 0.0, 0.0)


class MiniEvaQuarantinePolicy(ContactTracingPolicy):

    """Contact-tracing policy with minimal riskiness (family contacts only).

    Abandoned variant; prefer using a config file instead.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = RISK_FOR_LAYERS_MINI


class Exp2AQuarantinePolicy(ContactTracingPolicy):

    """Experimental variant A: full riskiness for family, school/work, and leisure.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 1.0, 1.0, 1.0, 0.0)


class Exp2BQuarantinePolicy(ContactTracingPolicy):

    """Experimental variant B: full riskiness for family and school/work only.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 1.0, 1.0, 0.0, 0.0)


class Exp2CQuarantinePolicy(ContactTracingPolicy):

    """Experimental variant C: full riskiness for family contacts only.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 1.0, 0.0, 0.0, 0.0)


class W10QuarantinePolicy(ContactTracingPolicy):

    """Contact-tracing policy with uniform 10 % riskiness across all groups.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.1, 0.1, 0.1)


class W20QuarantinePolicy(ContactTracingPolicy):

    """Contact-tracing policy with uniform 20 % riskiness across all groups.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.2, 0.2, 0.2)


class W30QuarantinePolicy(ContactTracingPolicy):

    """Contact-tracing policy with uniform 30 % riskiness across all groups.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.3, 0.3, 0.3)


class W40QuarantinePolicy(ContactTracingPolicy):

    """Contact-tracing policy with uniform 40 % riskiness across all groups.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.4, 0.4, 0.4)


class W60QuarantinePolicy(ContactTracingPolicy):

    """Contact-tracing policy with uniform 60 % riskiness across all groups.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.6, 0.6, 0.6)


class W80QuarantinePolicy(ContactTracingPolicy):

    """Contact-tracing policy with uniform 80 % riskiness across all groups.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
    """

    def __init__(self, graph, model):
        super().__init__(graph, model)
        self.riskiness = get_riskiness(self.graph, 0.8, 0.8, 0.8)


def _riskiness(contact, graph, riskiness):
    """Return the riskiness value for a given contact edge.

    Args:
        contact: Edge identifier (used to look up the layer).
        graph: The contact network graph object with a
            ``get_layer_for_edge`` method.
        riskiness (numpy.ndarray): Per-layer riskiness array.

    Returns:
        float: Riskiness value for the layer of the given contact edge.
    """
    #    print(f"DBG riskiness {graph.e_types[contact]}:{riskiness[graph.get_layer_for_edge(contact)]}")
    return riskiness[graph.get_layer_for_edge(contact)]
