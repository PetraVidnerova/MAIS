"""School epidemic-intervention policies for the MAIS simulation.

This module provides policies designed for school-specific contact
network graphs.  They manage week/weekend school-opening schedules,
optional rapid antigen testing, and class-level quarantine.

.. warning::
    These policies are intended to be run with a **special school
    graph** only.  Do not use them with general population graphs
    (e.g. hodoninsko, lounsko, papertown).

Classes:
    BasicSchoolPolicy: Manages school open/weekend toggle and optional
    student testing.
    ClosePartPolicy: Extends :class:`BasicSchoolPolicy` with the
    ability to close individual classes.
    AlternatingPolicy: Alternates two groups of classes each week.
    AlternateFreeMonday: Alternating policy with Monday as a free day.
    AlternateAndMondayPCR: Alternating policy with Monday PCR testing.
"""

# NOTE: this policy is intended to be run with a special graph for schools!!!!
# Do not use it for normal graphs (hodoninsko, lounsko, papertown, etc).
import global_configs as cfgs
from global_configs import monitor
from depo import Depo
import numpy as np
import pandas as pd
from policies.policy import Policy
from utils.history_utils import TimeSeries
import logging

from models.agent_based_network_model import STATES
from utils.config_utils import ConfigFile
from utils.graph_utils import compute_mean_degree

logging.basicConfig(level=logging.DEBUG)


class BasicSchoolPolicy(Policy):

    """Policy that manages school-day / weekend toggling and optional testing.

    On weekdays the school graph layers are active; on weekends all
    layers are switched off.  For the first 35 simulation steps all
    layers are also suppressed (warm-up period).  Optionally performs
    rapid antigen testing on configurable weekdays and places positive
    nodes into quarantine.

    Args:
        graph: The school contact network graph object.  Must expose
            ``num_nodes``, ``nodes_age``, ``layer_weights``,
            ``number_of_nodes``, ``QUARANTINE_COEFS``, ``nodes``,
            ``is_quarantined``, and ``nodes_class``.
        model: The epidemic model instance.
        config_file (str, optional): Path to an INI-style configuration
            file.  The ``[TESTING]`` section may contain:

            * ``testing`` – ``"Yes"`` to enable testing (default
              ``"No"``).
            * ``sensitivity`` – test sensitivity in [0, 1] (default
              0.4).
            * ``days`` – weekday index or list of indices on which
              testing is performed (default ``(0, 2)``).
        config_obj: Unused; reserved for future use.
    """

    def __init__(self, graph, model, config_file=None, config_obj=None):
        """Initialise the basic school policy.

        Args:
            graph: The school contact network graph object.
            model: The epidemic model instance.
            config_file (str, optional): Path to a configuration file.
            config_obj: Reserved for future use.
        """
        super().__init__(graph, model)

        self.weekend_start = 5
        self.weekend_end = 0

        self.first_day = True
        self.stopped = False
        self.testing = False
        self.test_sensitivity = 0.4
        self.test_days = (0, 2)
        self.test_groups = None

        self.at_school = np.ones(self.graph.num_nodes, dtype=bool)
        self.nodes = np.arange(self.graph.num_nodes)
        teachers = self.graph.nodes_age >= 20
        self.at_school[teachers] = 0  # do not test teachers

        self.cf = ConfigFile()
        if config_file is not None:
            self.cf.load(config_file)
            test_sec = self.cf.section_as_dict("TESTING")
            if test_sec.get("testing", "No") == "Yes":
                self.testing = True
            if "sensitivity" in test_sec:
                self.test_sensitivity = test_sec["sensitivity"]
            if "days" in test_sec:
                self.test_days = test_sec["days"]
                if not type(self.test_days) is list:
                    self.test_days = (self.test_days,)
                else:
                    self.test_days = [int(x) for x in self.test_days]

        logging.info(f"testing {self.testing}")
        logging.info(f"test sensitivity {self.test_sensitivity}")

        # all layers will be turned off for weekend
        self.mask_all_layers = {
            i: 0
            for i in range(len(self.graph.layer_weights))
        }
        self.back_up_layers = None

        # todo .. let it be a part of a graph
        # ZS:
        layers_apart_school = [5, 6, 12] + list(range(41, 72))
        #        layers_apart_school =  [2, 7, 11]
        self.school_layers = [
            x
            for x in range(len(self.graph.layer_weights))
            if x not in layers_apart_school
        ]

        self.positive_test = np.zeros(self.graph.num_nodes, dtype=bool)
        self.depo = Depo(self.graph.number_of_nodes)

        self.stat_in_quara = TimeSeries(301, dtype=int)

    def nodes_to_quarantine(self, nodes):
        """Remove nodes from school by switching off their school-layer edges.

        Sets ``at_school[nodes]`` to ``False`` and turns off all edges
        on school layers for those nodes.

        Args:
            nodes (numpy.ndarray): Indices of nodes to quarantine.
        """
        self.at_school[nodes] = False
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes:
            monitor(self.model.t,
                    "goes to the stay-at-home state.")
        edges_to_close = self.graph.get_nodes_edges_on_layers(
            nodes,
            self.school_layers
        )
        self.graph.switch_off_edges(edges_to_close)

    def nodes_from_quarantine(self, nodes):
        """Return nodes to school by switching on their school-layer edges.

        Sets ``at_school[nodes]`` to ``True`` and restores all edges on
        school layers for those nodes.

        Args:
            nodes (numpy.ndarray): Indices of nodes to release from
                quarantine.
        """
        self.at_school[nodes] = True

        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in nodes:
            monitor(self.model.t,
                    "goes to the go-to-school state.")
        edges_to_release = self.graph.get_nodes_edges_on_layers(
            nodes,
            self.school_layers
        )
        self.graph.switch_on_edges(edges_to_release)

    def first_day_setup(self):
        """Suppress all layers for the warm-up period and initialise statistics.

        Saves a copy of the initial layer weights, sets all layer
        weights to zero (school closed during warm-up), and fills
        ``stat_in_quara`` with zeros for days before the policy starts.
        """
        # # move teachers to R (just for one exp)
        # teachers = self.graph.nodes[self.graph.nodes_age >= 20]
        # #self.model.move_to_R(teachers)
        # self.nodes_to_quarantine(teachers)

        # switch off all layers till day 35
        # ! be careful about colision with layer calendar
        self.first_day_back_up = self.graph.layer_weights.copy()
        self.graph.set_layer_weights(self.mask_all_layers.values())
        self.stat_in_quara[0:self.model.t] = 0

    def do_testing(self):
        """Perform antigen testing on configured weekdays and quarantine positives.

        On test days, identifies students currently at school and
        stochastically classifies them as positive (with probability
        ``test_sensitivity``).  Positive nodes are quarantined for
        7 days.  Released nodes whose quarantine has ended are restored
        to school.  Updates ``stat_in_quara`` with the current
        quarantine count.
        """
        released = self.depo.tick_and_get_released()
        if len(released) > 0:
            self.graph.recover_edges_for_nodes(released)
            if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in released:
                monitor(self.model.t,
                        "node released from quarantine.")

        assert len(released) == 0 or self.model.t % 7 in self.test_days

        # monday or wednesday perform tests -> do it the night before
        if self.model.t % 7 in self.test_days:

            students_at_school = np.logical_and(
                self.at_school,
                self.graph.is_quarantined == 0
            )

            if self.test_groups is not None:
                should_not_be_tested = self.test_groups[self.test_passive]

                if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in should_not_be_tested:
                    monitor(self.model.t,
                            "node should NOT be tested.")

                students_at_school[should_not_be_tested] = False

            if cfgs.MONITOR_NODE is not None and students_at_school[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "node is at school and will be tested.")

            # at school and positive
            possibly_positive = (
                self.model.memberships[STATES.I_n] +
                self.model.memberships[STATES.I_s] +
                self.model.memberships[STATES.I_a] +
                self.model.memberships[STATES.J_n] +
                self.model.memberships[STATES.J_s]
            ).ravel()
            if cfgs.MONITOR_NODE is not None and possibly_positive[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "node is positive.")

            possibly_positive_test = np.logical_and(
                students_at_school,
                possibly_positive
            )
            if cfgs.MONITOR_NODE is not None and possibly_positive_test[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "node is positive and is tested.")

            self.positive_test.fill(0)
            num = possibly_positive_test.sum()
            r = np.random.rand(num)
            self.positive_test[possibly_positive_test] = r < self.test_sensitivity

            if cfgs.MONITOR_NODE is not None and self.positive_test[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "had positive tested.")

            self.depo.lock_up(self.positive_test, 7)
            self.graph.modify_layers_for_nodes(list(self.nodes[self.positive_test]),
                                               self.graph.QUARANTINE_COEFS)

            if cfgs.MONITOR_NODE is not None and self.graph.is_quarantined[cfgs.MONITOR_NODE]:
                monitor(self.model.t,
                        "is in quarantine.")

        self.stat_in_quara[self.model.t] = self.depo.num_of_prisoners

    def stop(self):
        """Signal the policy to stop any new interventions.

        After calling ``stop``, no new quarantines or school closures
        are initiated.
        """
        self.stopped = True

    def closing_and_opening(self):
        """Hook for subclasses to implement dynamic school-group opening/closing.

        Called each time-step after the weekend toggle.  The default
        implementation is a no-op.
        """
        pass

    def run(self):
        """Execute one time-step of the school policy.

        Handles first-day setup, weekend toggling, warm-up layer
        restoration (at day 35), subclass ``closing_and_opening`` hook,
        and optional testing (from day 35 onwards).
        """
        if self.first_day:
            self.first_day_setup()
            self.first_day = False

        logging.info(
            f"Hello world! This is the {self.__class__.__name__} function speaking.  {'(STOPPED)' if self.stopped else ''}")

        if self.model.t % 7 == self.weekend_start:
            logging.info("Start weekend, closing.")
            self.back_up_layers = self.graph.layer_weights.copy()
            self.graph.set_layer_weights(self.mask_all_layers.values())

        if self.model.t % 7 == self.weekend_end:
            logging.info("End weekend, opening.")
            if self.back_up_layers is None:
                logging.warning("The school policy started during weekend!")
            else:
                self.graph.set_layer_weights(self.back_up_layers)

        if (self.first_day_back_up is not None and self.model.t >= 35
                and self.model.t % 7 == self.weekend_end):  # 35 is sunday! run it after end of weekend
            self.graph.set_layer_weights(self.first_day_back_up)
            self.first_day_back_up = None  # run it only once
            # print(f"t={self.model.t}")
            # print(self.graph.layer_weights)
            # exit()

        self.closing_and_opening()

        if self.model.t >= 35 and self.testing:
            self.do_testing()

        # if self.model.t % 7 == 1:
        #    # print every week the mean degree of second group
        #    students = self.graph.nodes[self.graph.nodes_age < 20]
        #    mean_degree = compute_mean_degree(self.graph, students)
        #    logging.debug(f"Day {self.model.t}: Mean degree of a student {mean_degree}")

    def to_df(self):
        """Return a DataFrame with daily school-quarantine statistics.

        Returns:
            pandas.DataFrame: DataFrame indexed by time ``T`` with
            column ``school_policy_in_quara`` (number of nodes in
            school quarantine) and ``day``.
        """
        index = range(0, self.model.t+1)
        columns = {
            f"school_policy_in_quara":  self.stat_in_quara[:self.model.t+1],
        }
        columns["day"] = np.floor(index).astype(int)
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        return df


class ClosePartPolicy(BasicSchoolPolicy):

    """School policy that can close specific classes listed in a config file.

    Extends :class:`BasicSchoolPolicy` with helper methods to quarantine
    or release all nodes belonging to a named set of classes.  On
    first-day setup the classes listed under ``[CLOSED]`` in the config
    file are sent to quarantine, and optionally all teacher edges are
    closed.

    Args:
        graph: The school contact network graph object.
        model: The epidemic model instance.
        config_file (str, optional): Path to a configuration file.
            The ``[CLOSED]`` section may contain:

            * ``close_teachers`` – ``"Yes"`` to close teacher edges
              (default ``"No"``).
            * ``classes`` – list of class names to quarantine at start.
        config_obj: Reserved for future use.
    """

    def convert_class(self, a):
        """Convert node class indices to class-name strings.

        Args:
            a (numpy.ndarray): Array of integer class indices.

        Returns:
            numpy.ndarray: Array of class-name strings (or ``None``
            for out-of-range indices).
        """
        _convert = np.vectorize(lambda x: (
            self.graph.cat_table["class"]+[None])[x])
        return _convert(a)

    def nodes_in_classes(self, list_of_classes):
        """Return node indices belonging to any of the specified classes.

        Args:
            list_of_classes (list[str]): Class names to look up.

        Returns:
            numpy.ndarray: Indices of all nodes whose class name is in
            ``list_of_classes``.
        """
        # todo - save node_classes? not to convert every time
        node_classes = self.convert_class(self.graph.nodes_class)
        return self.graph.nodes[np.isin(node_classes, list_of_classes)]

    def classes_to_quarantine(self, list_of_classes):
        """Send all nodes in the specified classes to quarantine.

        Marks nodes as not at school and switches off their school-layer
        edges.

        Args:
            list_of_classes (list[str]): Class names whose members
                should be quarantined.
        """
        self.nodes_to_close = self.nodes_in_classes(list_of_classes)
        self.at_school[self.nodes_to_close] = False
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in self.nodes_to_close:
            monitor(self.model.t,
                    "goes to the stay-at-home state.")

        # self.graph.modify_layers_for_nodes(self.nodes_to_close,
        #                                   self.mask_all_layers)
        edges_to_close = self.graph.get_nodes_edges_on_layers(
            self.nodes_to_close,
            self.school_layers
        )
        self.graph.switch_off_edges(edges_to_close)

    def classes_from_quarantine(self, list_of_classes):
        """Release all nodes in the specified classes from quarantine.

        Marks nodes as at school and switches on their school-layer
        edges.

        Args:
            list_of_classes (list[str]): Class names whose members
                should be released.
        """
        self.nodes_to_release = self.nodes_in_classes(list_of_classes)
        #        self.graph.recover_edges_for_nodes(self.nodes_to_release)
        self.at_school[self.nodes_to_release] = True
        if cfgs.MONITOR_NODE is not None and cfgs.MONITOR_NODE in self.nodes_to_release:
            monitor(self.model.t,
                    "goes to the go-to-school state.")
        edges_to_release = self.graph.get_nodes_edges_on_layers(
            self.nodes_to_release,
            self.school_layers
        )
        self.graph.switch_on_edges(edges_to_release)

    def first_day_setup(self):
        """Run parent first-day setup and apply initial class closures from config.

        Optionally closes teacher edges and quarantines classes listed
        under ``[CLOSED]`` in the configuration file.
        """
        super().first_day_setup()

        close_teachers = self.cf.section_as_dict(
            "CLOSED").get("close_teachers", "No")
        if close_teachers == "Yes":
            teachers = self.graph.nodes[self.graph.nodes_age >= 20]
            edges_to_close = self.graph.get_nodes_edges_on_layers(
                teachers,
                self.school_layers
            )
            self.graph.switch_off_edges(edges_to_close)

        # move teachers to R (just for one exp)
        #teachers = self.graph.nodes[self.graph.nodes_age >= 20]
        # self.model.move_to_R(teachers)

        # classes listed in config file goes to quarantine
        classes_to_close = self.cf.section_as_dict(
            "CLOSED").get("classes", list())
        if len(classes_to_close) > 0:
            logging.info(f"Closing classes {classes_to_close}")
            self.classes_to_quarantine(classes_to_close)
        else:
            logging.info("No classes clossed.")


class AlternatingPolicy(ClosePartPolicy):

    """School policy that alternates two groups of classes week by week.

    One group attends school while the other stays home; the groups
    swap every week (at ``weekend_end``).  Optionally, testing sub-groups
    can be defined to alternate which half of each group is tested on a
    given day.

    The groups are either defined explicitly in the config file
    (``[ALTERNATE]`` section with ``group1`` and ``group2`` class lists)
    or derived from the graph's ``nodes_class_group`` attribute when
    ``use_class_groups = Yes``.

    Args:
        graph: The school contact network graph object.
        model: The epidemic model instance.
        config_file (str, optional): Path to a configuration file with
            ``[ALTERNATE]`` and optionally ``[TESTING_GROUPS]``
            sections.
    """

    def __init__(self, graph, model, config_file=None):
        """Initialise the alternating policy with group definitions from config.

        Args:
            graph: The school contact network graph object.
            model: The epidemic model instance.
            config_file (str, optional): Path to a configuration file.
        """
        super().__init__(graph, model, config_file)

        rotate_class_groups = self.cf.section_as_dict(
            "ALTERNATE").get("use_class_groups", "No") == "Yes"

        if not rotate_class_groups:
            group1 = self.cf.section_as_dict(
                "ALTERNATE").get("group1", list())
            group2 = self.cf.section_as_dict(
                "ALTERNATE").get("group2", list())

            nodes_group1 = self.nodes_in_classes(group1)
            nodes_group2 = self.nodes_in_classes(group2)
            self.groups = (nodes_group1, nodes_group2)
        else:
            nodes_group1 = self.graph.nodes[self.graph.nodes_class_group == 0]
            nodes_group2 = self.graph.nodes[self.graph.nodes_class_group == 1]

            self.groups = (nodes_group1, nodes_group2)

        self.passive_group, self.active_group = 1, 0
        self.nodes_to_quarantine(self.groups[self.passive_group])

        testing_cfg = self.cf.section_as_dict("TESTING_GROUPS")
        if testing_cfg:
            use_class_groups = testing_cfg.get(
                "use_class_groups", "No") == "Yes"

            if not use_class_groups:
                group1a = testing_cfg["group1a"]
                group1b = testing_cfg["group1b"]
                group2a = testing_cfg["group2a"]
                group2b = testing_cfg["group2b"]

                groupA = self.nodes_in_classes(group1a+group2a)
                groupB = self.nodes_in_classes(group1b+group2b)

                self.test_groups = (groupA, groupB)
                self.test_passive, self.test_active = 0, 1

            else:
                groupA = self.graph.nodes[self.graph.nodes_class_group == 0]
                groupB = self.graph.nodes[self.graph.nodes_class_group == 1]

                self.test_groups = (groupA, groupB)
                self.test_passive, self.test_active = 0, 1
        else:
            self.test_groups = None

    def closing_and_opening(self):
        """Alternate active and passive groups at the start of each school week.

        At ``weekend_end`` the active and passive groups are swapped:
        the previously active group is quarantined and the previously
        passive group is released.  Every two weeks the testing sub-groups
        are also rotated.
        """

        if self.model.t % 7 == self.weekend_end:
            self.passive_group, self.active_group = self.active_group, self.passive_group

            self.nodes_from_quarantine(self.groups[self.active_group])
            self.nodes_to_quarantine(self.groups[self.passive_group])

            logging.info(f"Day {self.model.t}: Groups changed. Active group is {self.active_group}")

        if self.model.t % 14 == self.weekend_end:
            if self.test_groups is not None:
                self.test_active, self.test_passive = self.test_passive, self.test_active

        # if self.model.t % 7 == 1:
        #    # print every week the mean degree of second group
        #    group2 = self._nodes_in_classes(self.groups[1])
        #    mean_degree = compute_mean_degree(self.graph, group2)
        #    logging.debug(f"Day {self.model.t}: Mean degree of group2 {mean_degree}")


class AlternateFreeMonday(AlternatingPolicy):

    """Alternating policy where Monday is a free day (school starts Tuesday).

    Groups alternate weekly.  Testing is disabled by default.

    Args:
        graph: The school contact network graph object.
        model: The epidemic model instance.
        config_file (str, optional): Path to a configuration file.
    """

    def __init__(self, graph, model, config_file=None):
        """Initialise with Monday as the first school day and no testing.

        Args:
            graph: The school contact network graph object.
            model: The epidemic model instance.
            config_file (str, optional): Path to a configuration file.
        """
        super().__init__(graph, model, config_file)

        self.weekend_start = 5
        self.weekend_end = 1
        self.testing = False


class AlternateAndMondayPCR(AlternatingPolicy):

    """Alternating policy with high-sensitivity (PCR-equivalent) Monday testing.

    Groups alternate weekly.  Testing is enabled on Mondays (weekday
    index 1) with a sensitivity of 0.8 (mimicking PCR).

    Args:
        graph: The school contact network graph object.
        model: The epidemic model instance.
        config_file (str, optional): Path to a configuration file.
    """

    def __init__(self, graph, model, config_file=None):
        """Initialise with Monday testing at 80 % sensitivity.

        Args:
            graph: The school contact network graph object.
            model: The epidemic model instance.
            config_file (str, optional): Path to a configuration file.
        """
        super().__init__(graph, model, config_file)

        self.weekend_start = 5
        self.weekend_end = 1

        self.testing = True
        self.test_sensitivity = 0.8
        self.test_days = (1,)
