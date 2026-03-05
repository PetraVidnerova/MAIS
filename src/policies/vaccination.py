"""Vaccination policies for the MAIS epidemic simulation.

This module provides several vaccination policy classes that vaccinate
elderly and worker sub-populations according to a configurable calendar.
Different subclasses implement distinct mechanisms by which vaccination
reduces infection risk:

* :class:`Vaccination` – on first exposure (entering state E) a
  vaccinated node may be redirected back to susceptible.
* :class:`VaccinationToR` – vaccinated nodes in S are moved directly
  to R (recovered/immune).
* :class:`VaccinationToA` – vaccination increases the asymptomatic
  rate.
* :class:`VaccinationToSA` – combines the ``ToS`` and ``ToA``
  mechanisms.
"""

import time
import numpy as np
import pandas as pd
from policies.policy import Policy
from utils.history_utils import TimeSeries
import logging

from models.agent_based_network_model import STATES
from utils.config_utils import ConfigFile

logging.basicConfig(level=logging.DEBUG)


def _process_calendar(filename):
    """Load a vaccination calendar CSV and return per-group daily counts.

    The CSV must have columns ``T`` (simulation day), ``workers``
    (number of workers to vaccinate), and ``elderly`` (number of
    elderly individuals to vaccinate).

    Args:
        filename (str): Path to the CSV vaccination calendar file.

    Returns:
        tuple[dict, dict]: A pair ``(workers_calendar, elderly_calendar)``
        where each is a ``{day: count}`` dictionary.
    """
    df = pd.read_csv(filename)
    return (
        dict(zip(df["T"], df["workers"].astype(int))),
        dict(zip(df["T"], df["elderly"].astype(int))),
    )


class Vaccination(Policy):

    """Vaccination policy that reduces susceptibility upon exposure.

    When a vaccinated node first enters state ``E`` (exposed), there is
    a probability (dependent on days since vaccination and whether a
    first or second dose has been given) that the node is returned to
    the susceptible state instead of progressing towards illness.

    Vaccination is administered daily according to separate calendars
    for elderly and worker sub-populations.  The number of days since
    vaccination is tracked per node.

    Args:
        graph: The contact network graph object.  Must expose
            ``num_nodes``, ``nodes_age``, ``nodes_ecactivity``, and
            ``cat_table``.
        model: The epidemic model instance.
        config_file (str): Path to an INI-style configuration file.
            **Required.**  The file must include:

            * ``[CALENDAR]`` section with ``calendar_filename`` (path
              to a vaccination calendar CSV) and optionally ``delay``
              (days between first and second dose).
            * ``[EFFECT]`` section with ``first_shot`` and
              ``second_shot`` effectiveness coefficients.

    Raises:
        str: If ``config_file`` is ``None`` or the calendar filename
            is missing (raises a string literal – legacy behaviour).
    """

    def __init__(self, graph, model, config_file=None):
        """Initialise the vaccination policy from a configuration file.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
            config_file (str): Path to the required configuration file.
        """
        super().__init__(graph, model)

        self.first_day = True
        self.stopped = False
        self.delay = None

        # -1 .. not vaccinated
        # >= 0 days from vaccination
        self.vaccinated = np.full(
            self.graph.num_nodes, fill_value=-1, dtype=int)
        self.nodes = np.arange(self.graph.num_nodes)
        self.days_in_E = np.zeros(self.graph.num_nodes,  dtype=int)
        self.target_for_R = np.zeros(
            self.graph.num_nodes, dtype=bool)  # auxiliary var

        # statistics
        self.stat_moved_to_R = TimeSeries(401, dtype=int)

        if config_file:
            cf = ConfigFile()
            cf.load(config_file)
            calendar_filename = cf.section_as_dict(
                "CALENDAR").get("calendar_filename", None)
            if calendar_filename is None:
                raise "Missing calendar filename in vaccination policy config file."
            self.workers_calendar, self.elderly_calendar = _process_calendar(
                calendar_filename)
            self.delay = cf.section_as_dict("CALENDAR").get("delay", None)

            self.first_shot_coef = cf.section_as_dict("EFFECT")["first_shot"]
            self.second_shot_coef = cf.section_as_dict("EFFECT")["second_shot"]
        else:
            raise "Vaccination policy requires config file."

        self.old_to_vaccinate = list(np.argsort(self.graph.nodes_age))
        # self.index_to_go = len(self.sort_indicies)-1

        worker_id = self.graph.cat_table["ecactivity"].index("working")
        self.workers_to_vaccinate = list(
            self.nodes[self.graph.nodes_ecactivity == worker_id])
        # print(self.workers_to_vaccinate)
        # exit()

    def first_day_setup(self):
        """Perform first-day setup (no-op for this policy)."""
        pass

    def stop(self):
        """Signal the policy to stop vaccinating new nodes.

        After calling ``stop``, nodes already being tracked continue to
        have their vaccination days incremented, but no new vaccinations
        are administered.
        """
        self.stopped = True

    def move_to_S(self):
        """Redirect newly exposed vaccinated nodes back to the susceptible state.

        For each vaccinated node that has just entered state ``E`` for
        the first time today, a random draw determines whether the
        vaccine prevents progression.  The probability depends on
        days-since-vaccination:

        * 14 to (``delay`` + 6) days: ``first_shot_coef``.
        * (``delay`` + 7) days or more: ``second_shot_coef``.

        Nodes redirected to susceptible have their ``days_in_E`` counter
        reset.  The count of redirected nodes is recorded in
        ``stat_moved_to_R``.
        """
        # take those who are first day E (are E AND are E the first day)
        nodes_first_E = (self.model.memberships[STATES.E] == 1).ravel()
        self.days_in_E[nodes_first_E] += 1
        nodes_first_E = np.logical_and(
            nodes_first_E,
            self.days_in_E == 1
        )

        if nodes_first_E.sum() == 0:
            return

        # By 14 days after the first shot, the effect is zero (i.e.  an
        # infectedindividual becomes exposed and later symptomatic or asymptomaticas if
        # not vaccinated)•Between 14 and 20 days after the first shot, those, who are
        # infected(heading to theEcompartment) and are "intended" to be
        # asymptomatic(further go toIa, it is no harm to assume this decision is made
        # inforward) become recovered with probability0.29instead of
        # enteringtheEcompartment. Those, intended to be symptomatic (further gotoIp)
        # become recovered with0.46probability.•21days or more after first shot, this
        # probability of "recovery" is0.52for asymptomatic and0.6for symptomatic.•7days
        # after the second shot or later, the probability of "recovery" is0.9for
        # asymptomatic and0.92for symptomatic

        # divide nodes_first_E to asymptomatic candidates and symptomatic candidates
        # assert np.all(np.logical_or(
        #     self.model.state_to_go[nodes_first_E, 0] == STATES.I_n,
        #     self.model.state_to_go[nodes_first_E, 0] == STATES.I_a
        # )), "inconsistent state_to_go"

        self.target_for_R.fill(0)

        def decide_move_to_R(selected, prob):
            n = len(selected)
            print(f"generating {n} randoms")
            if n > 0:
                r = np.random.rand(n)
                self.target_for_R[selected] = r < prob

        # 14 - 20 days: 0.29 for A, 0.46 for S
        # skip those with < 14 days

        # for state, probs in (
        #         (STATES.I_n, [0.29, 0.52, 0.9]),
        #         (STATES.I_a, [0.46, 0.6, 0.92])
        # ):
        #     nodes_heading_to_state = nodes_first_E.copy()
        #     nodes_heading_to_state[nodes_first_E] = self.model.state_to_go[nodes_first_E, 0] == state
        #     node_list = self.nodes[nodes_heading_to_state]

        #     if not(len(node_list) > 0):
        #         continue
        #     # skip those who are in first 14 days
        #     node_list = node_list[self.vaccinated[node_list] >= 14]
        #     # select 14 - 21
        #     selected = node_list[self.vaccinated[node_list] < 21]
        #     decide_move_to_R(selected, probs[0])
        #     # skip them
        #     node_list = node_list[self.vaccinated[node_list] >= 21]
        #     # selecte < second shot + 7
        #     selected = node_list[self.vaccinated[node_list] < self.delay + 7]
        #     decide_move_to_R(selected, probs[1])
        #     # skip them
        #     node_list = node_list[self.vaccinated[node_list] >= self.delay + 7]
        #     decide_move_to_R(node_list, probs[2])

        # first shots

        node_list = self.nodes[nodes_first_E]

        if not(len(node_list) > 0):
            return

        # those who have only the first shot
        first_shotters = node_list[
            np.logical_and(
                self.vaccinated[node_list] >= 14,
                self.vaccinated[node_list] < self.delay + 7
            )]
        r = np.random.rand(len(first_shotters))
        go_back = first_shotters[r < self.first_shot_coef]
        self.target_for_R[go_back] = True

        second_shotters = node_list[self.vaccinated[node_list]
                                    >= self.delay + 7]
        r = np.random.rand(len(second_shotters))
        go_back = second_shotters[r < self.second_shot_coef]
        self.target_for_R[go_back] = True

        self.stat_moved_to_R[self.model.t] = self.target_for_R.sum()
        self.model.move_target_nodes_to_S(self.target_for_R)
        self.days_in_E[self.target_for_R] = 0

    def process_vaccinated(self):
        """Apply the vaccination effect to currently vaccinated nodes.

        Calls :meth:`move_to_S` to handle newly exposed vaccinated
        nodes.  Subclasses override this method to implement alternative
        vaccination mechanisms.
        """
        self.move_to_S()

    def run(self):
        """Execute one time-step of the vaccination policy.

        Increments vaccination-day counters, applies the vaccination
        effect, and administers new vaccinations according to the
        daily calendar.
        """
        super().run()

        # update vaccinated days
        already_vaccinated = self.vaccinated != -1
        self.vaccinated[already_vaccinated] += 1

        self.process_vaccinated()

        # update asymptotic rates  - OBSOLETE
        # Počítám, že první týden nemá vakcíná
        # žádnou účinnost, po týdnu 50%, po dvou týdnech 70%, po druhé
        # dávce 90% a po dalším týdnu 95%

        # older = self.graph.nodes_age > 65
        # younger = np.logical_not(older)

        # # update two weeks after first vaccination
        # selected = self.vaccinated == 14
        # self.model.asymptomatic_rate[np.logical_and(selected, older)] = 0.7
        # self.model.asymptomatic_rate[np.logical_and(selected, younger)] = 0.9

        # # update two weeks after second vaccination
        # selected = self.vaccinated == self.delay + 14
        # self.model.asymptomatic_rate[np.logical_and(selected, older)] = 0.8
        # self.model.asymptomatic_rate[np.logical_and(selected, younger)] = 0.95

        # selected = self.vaccinated == 7
        # self.model.asymptomatic_rate[selected] = 0.5
        # selected = self.vaccinated == 14
        # self.model.asymptomatic_rate[selected] = 0.7
        # selected = self.vaccinated == self.delay
        # self.model.asymptomatic_rate[selected] = 0.9
        # selected = self.vaccinated == self.delay + 7
        # self.model.asymptomatic_rate[selected] = 0.95

        logging.debug(f"asymptomatic rate {self.model.asymptomatic_rate.mean()}")

        if self.model.T in self.elderly_calendar:
            self.vaccinate_old(self.elderly_calendar[self.model.T])

        if self.model.T in self.workers_calendar:
            self.vaccinate_workers(self.workers_calendar[self.model.T])

    def vaccinate_old(self, num):
        """Vaccinate up to ``num`` elderly nodes (sorted by descending age).

        Nodes that are already vaccinated, currently detected as active
        cases, or dead are skipped.

        Args:
            num (int): Maximum number of elderly nodes to vaccinate today.
        """
        if num == 0:
            return
        logging.info(f"T={self.model.T} Vaccinating {num} elderly.")
        index = len(self.old_to_vaccinate)
        while num > 0 and index > 0:
            index -= 1
            who = self.old_to_vaccinate[index]
            if self.vaccinated[who] != -1:
                continue
            if self.model.node_detected[who]:  # change to active case
                continue
            # dead are not vaccinated
            if self.model.memberships[STATES.D, who, 0] == 1:
                continue
            self.vaccinated[who] = 0
            del self.old_to_vaccinate[index]
            num -= 1

    def vaccinate_workers(self, num):
        """Vaccinate up to ``num`` worker nodes chosen at random.

        Nodes that are currently detected as active cases or dead are
        excluded from selection.  If fewer eligible workers than ``num``
        exist, all eligible workers are vaccinated.

        Args:
            num (int): Target number of workers to vaccinate today.
        """
        if num == 0:
            return
        logging.info(f"T={self.model.T} Vaccinating {num} workers.")
        num_workers = len(self.workers_to_vaccinate)
        if num_workers == 0:
            return

        # ids_to_vaccinate = self.workers_to_vaccinate[self.model.node_detected[self.workers_to_vaccinate] == False]
        # if len(ids_to_vaccinate) == 0:
        #     logging.warning("No more workers to vaccinate.")
        #     exit()
        #     return
        # ids_to_vaccinate = ids_to_vaccinate[self.model.memberships[STATES.D, ids_to_vaccinate, 0] != 1]

        ids_to_vaccinate = np.logical_and(
            self.model.node_detected[self.workers_to_vaccinate] == False,
            self.model.memberships[STATES.D, self.workers_to_vaccinate, 0] != 1
        ).nonzero()[0]

        if len(ids_to_vaccinate) < num:
            logging.info("Not enough workers to vaccinate.")
            num = len(ids_to_vaccinate)
            if num == 0:
                return
        selected_ids = np.random.choice(
            ids_to_vaccinate, size=num, replace=False)
        for index in selected_ids:
            who = self.workers_to_vaccinate[index]
            self.vaccinated[who] = 0
        for index in sorted(selected_ids, reverse=True):
            del self.workers_to_vaccinate[index]

        # # get all nodes that are S or Ss and were not vaccinated
        # target_nodes = np.logical_not(
        #     self.model.node_detected
        # )
        # target_nodes = np.logical_and(
        #     target_nodes[:,0],
        #     self.vaccinated == False
        # )
        # print(target_nodes.shape)
        # pool = self.nodes[target_nodes]

        # # select X of them to be vaccinated
        # to_vaccinate = np.random.choice(pool, size=self.num_to_vaccinate, replace=False)
        # self.vaccinated[to_vaccinate] = True
        # self.model.asymptomatic_rate[to_vaccinate] = 0.9

        # #        self.model.move_to_R(to_vaccinate)

    def to_df(self):
        """Return a DataFrame with daily vaccination statistics.

        Returns:
            pandas.DataFrame: DataFrame indexed by time ``T`` with
            column ``moved_to_R`` (nodes redirected to susceptible/
            recovered each day) and ``day``.
        """
        index = range(0+self.model.start_day-1, self.model.t +
                      self.model.start_day)  # -1 + 1
        policy_name = type(self).__name__
        columns = {
            f"moved_to_R": self.stat_moved_to_R[:self.model.t+1],
        }
        columns["day"] = np.floor(index).astype(int)
        df = pd.DataFrame(columns, index=index)
        df.index.rename('T', inplace=True)
        return df


class VaccinationToR(Vaccination):

    """Vaccination policy that moves susceptible nodes directly to recovered.

    On day 14 after the first shot, susceptible vaccinated nodes are
    moved to R with probability ``first_shot_coef``.  On day
    ``delay + 7`` after the first shot, a further fraction
    ``(second_shot_coef - first_shot_coef)`` is moved to R.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
        config_file (str): Path to the required configuration file.
    """

    def process_vaccinated(self):
        """Move eligible susceptible vaccinated nodes to recovered state.

        Applies first-shot and second-shot effects by drawing random
        numbers and calling ``model.move_target_nodes_to_R``.
        """
        # # update two weeks after first vaccination
        nodes_in_S = self.nodes[self.model.memberships[STATES.S, :, 0] == 1]

        selected = nodes_in_S[self.vaccinated[nodes_in_S] == 14]
        r = np.random.rand(len(selected))
        to_R = selected[r < self.first_shot_coef]

        self.target_for_R.fill(0)
        self.target_for_R[to_R] = True

        selected = nodes_in_S[self.vaccinated[nodes_in_S] == self.delay + 7]
        r = np.random.rand(len(selected))
        to_R = selected[r < (self.second_shot_coef - self.first_shot_coef)]
        self.target_for_R[to_R] = True

        self.stat_moved_to_R[self.model.t] = self.target_for_R.sum()
        self.model.move_target_nodes_to_R(self.target_for_R)
        self.days_in_E[self.target_for_R] = 0


class VaccinationToA(Vaccination):

    """Vaccination policy that increases the asymptomatic rate of vaccinated nodes.

    Fourteen days after the first shot the asymptomatic rate is
    updated to reflect first-dose effectiveness.  A further update
    occurs at ``delay + 7`` days to reflect second-dose effectiveness.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
        config_file (str): Path to the required configuration file.
    """

    def update_asymptomatic_rates(self):
        """Update ``model.asymptomatic_rate`` for nodes at key vaccination milestones.

        First-shot effect is applied at day 14; second-shot effect at
        day ``self.delay + 7``.
        """
        # # update two weeks after first vaccination
        selected = self.nodes[self.vaccinated == 14]
        srate = 1 - 0.179
        self.model.asymptomatic_rate[selected] = 1 - \
            srate*(1-self.first_shot_coef)

        selected = self.nodes[self.vaccinated == self.delay + 7]
        self.model.asymptomatic_rate[selected] = 1 - \
            srate*(1-self.second_shot_coef)

    def process_vaccinated(self):
        """Apply vaccination effect by updating asymptomatic rates."""
        self.update_asymptomatic_rates()


class VaccinationToSA(VaccinationToA):

    """Vaccination policy combining susceptible redirection and asymptomatic-rate update.

    Applies both the :meth:`Vaccination.move_to_S` mechanism (redirecting
    newly exposed vaccinated nodes back to susceptible) and the
    :meth:`VaccinationToA.update_asymptomatic_rates` mechanism each
    time-step.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
        config_file (str): Path to the required configuration file.
    """

    def process_vaccinated(self):
        """Apply both susceptible-redirection and asymptomatic-rate-update effects."""
        self.move_to_S()
        self.update_asymptomatic_rates()
