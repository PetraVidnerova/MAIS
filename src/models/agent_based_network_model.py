"""Agent-based epidemic network model (SimulationDrivenModel).

This module defines :class:`SimulationDrivenModel`, the primary agent-based
epidemiological model of the MAIS project.  It inherits from
:class:`~models.simulation_engine.SimulationEngine` and implements:

* Stochastic duration sampling for every disease stage using pre-loaded
  probability distributions (JSON file).
* Age- and sex-stratified mortality via a CSV death-probability table.
* Network-contact infection via :func:`~models.prob_infection.prob_of_contact`.
* Support for external (EXT) nodes with controllable edge activity.
* Utility methods for scenario manipulation (``move_to_E``, ``move_to_R``,
  etc.) and output statistics (``df_source_infection``,
  ``df_source_nodes``).
"""

import json
import numpy as np
import pandas as pd

import time
import logging

from models.simulation_engine import SimulationEngine
from models.prob_infection import prob_of_contact
from utils.random_utils import RandomDuration
from utils.random_utils import gen_tuple
from utils.history_utils import TimeSeries, TransitionHistory, ShortListSeries
from models.states import STATES, state_codes

from utils.global_configs import monitor
import utils.global_configs as global_configs


class SimulationDrivenModel(SimulationEngine):
    """Agent-based SEIR-variant epidemic model with stochastic duration sampling.

    Each agent (network node) progresses through disease compartments with
    durations drawn from empirical distributions.  Key features:

    * Compartments: S, S_s, E, I_n, I_a, I_s, J_s, J_n, R, D, EXT.
    * Duration distributions loaded from a JSON file at initialisation.
    * Age/sex-stratified case-fatality rates from a CSV file.
    * Daily infection driven by
      :func:`~models.prob_infection.prob_of_contact`.
    * Optional external-node mechanism (EXT state) with probabilistic edge
      activation.

    Class-level attributes define the model structure (states, transitions,
    parameters).  All are overridable by subclasses.
    """

    states = [
        STATES.S,
        STATES.S_s,
        STATES.E,
        STATES.I_n,
        STATES.I_a,
        STATES.I_s,
        STATES.J_s,
        STATES.J_n,
        STATES.R,
        STATES.D,
        STATES.EXT
    ]

    num_states = len(states)
    state_str_dict = state_codes
    ext_code = STATES.EXT

    
    transitions = [
        (STATES.S_s,  STATES.E),  # 0
        (STATES.S_s,  STATES.S),  # 1

        (STATES.S, STATES.E),  # 3
        (STATES.S, STATES.S_s),  # 4

        (STATES.E, STATES.I_n),  # 7
        (STATES.E, STATES.I_a),  # 8

        (STATES.I_n, STATES.J_n),
        (STATES.I_a, STATES.I_s),
        (STATES.I_s, STATES.J_s),

        (STATES.J_s, STATES.R),
        (STATES.J_s, STATES.D),

        (STATES.J_n, STATES.R)
    ]

    num_transitions = len(transitions)

    final_states = [
        STATES.R,
        STATES.D
    ]

    invisible_states = [
        STATES.D,
        STATES.EXT
    ]

    unstable_states = [
        STATES.E,
        STATES.I_n,
        STATES.I_a,
        STATES.I_s,
        STATES.J_n,
        STATES.J_s
    ]

    fixed_model_parameters = {
        "p": (0, "probability of interaction outside adjacent nodes"),
        "q": (0, " probability of detected individuals interaction outside adjacent nodes"),
        "mu": (0, "rate of infection-related death"),
        "false_symptoms_rate": (0, ""),
        "false_symptoms_recovery_rate": (1., ""),
        "save_nodes": (False, ""),
        "durations_file": ("../config/duration_probs.json", "file with probs for durations"),
        "prob_death_file": ("../data/prob_death.csv", "file with probs for durations"),
        "ext_epi": (0, "prob of beeing infectious for external nodes"),
        "beta_reduction": (0,  "reduction of beta for asymptomatic multiplier")
    }

    model_parameters = {
        "beta": (0,  "rate of transmission (exposure)"),
        "beta_in_family": (0, "hidden parameter"),
        "beta_A": (0, "hidden parameter"),
        "beta_A_in_family": (0, "hidden parameter"),
        "theta_E": (0, "rate of baseline testing for exposed individuals"),
        "theta_Ia": (0, "rate of baseline testing for Ia individuals"),
        "theta_Is": (0, "rate of baseline testing for Is individuals"),
        "theta_In": (0, "rate of baseline testing for In individuals"),
        "test_rate": (1.0, "test rate"),
        "psi_E": (0, "probability of positive test results for exposed individuals"),
        "psi_Ia": (0, "probability of positive test results for Ia individuals"),
        "psi_Is": (0, "probability of positive test results for Is individuals"),
        "psi_In": (0, "probability of positive test results for In individuals"),
        "asymptomatic_rate": (0, "asymptomatic rate"),
        "symptomatic_time": (-1, "time_from first_symptom  - do not setup"),
        "infectious_time": (-1, "time_from first_symptom  - do not setup"),
    }




    def inicialization(self):
        """Initialise the model: set derived beta parameters, load duration and death data.

        Before calling the parent initialiser, sets ``beta_in_family``,
        ``beta_A``, and ``beta_A_in_family`` in ``init_kwargs`` based on the
        supplied ``beta`` and ``beta_reduction``.

        After parent init:

        * Allocates ``self.testable`` and ``self.will_die`` arrays.
        * Loads duration-probability distributions from ``self.durations_file``
          (JSON).
        * Loads age/sex death probabilities from ``self.prob_death_file``
          (CSV).
        """
        self.init_kwargs["beta_in_family"] = self.init_kwargs["beta"]
        self.init_kwargs["beta_A"] = self.init_kwargs["beta"] * \
            self.init_kwargs["beta_reduction"]
        self.init_kwargs["beta_A_in_family"] = self.init_kwargs["beta_A"]

        super().inicialization()

        self.testable = np.zeros(
            shape=(self.graph.number_of_nodes, 1), dtype=bool)

        self.will_die = np.zeros(
            shape=(self.graph.number_of_nodes,), dtype=bool)

        # initialize random generators for durations
        with open(self.durations_file, "r") as f:
            self.duration_probs = json.load(f)

        self.rngd = {
            label: RandomDuration(probs)
            for label, probs in self.duration_probs.items()
        }

        # load death probs
        # first get codes for M an F
        self.MALE = self.graph.cat_table["sex"].index("M")
        self.FEMALE = self.graph.cat_table["sex"].index("F")

        # read and convert to dictionary for better lookup
        df = pd.read_csv(self.prob_death_file)
        df = df.set_index("age")
        df.rename(columns={"F": self.FEMALE,
                           "M": self.MALE},
                  inplace=True)
        self.death_probs = {
            self.FEMALE: df[self.FEMALE].to_numpy(),
            self.MALE: df[self.MALE].to_numpy()
        }


    def setup_series_and_time_keeping(self):
        """Extend time-keeping setup with infection-time and contact-history buffers.

        Adds:

        * ``self.infect_time`` – per-node infectious-period counter.
        * ``self.contact_history`` – 14-day rolling contact buffer.
        * ``self.successfull_source_of_infection`` – per-node count of
          successful transmissions originated.
        * ``self.stat_successfull_layers`` – per-layer daily transmission
          counts.
        """
        super().setup_series_and_time_keeping()

        self.infect_time = np.zeros(self.num_nodes, dtype=int)

        # history of contacts for last 14 days
        self.contact_history = ShortListSeries(14)
        for i in range(14):
            self.contact_history.append(None)
        self.successfull_source_of_infection = np.zeros(
            self.num_nodes, dtype="uint16")

        self.stat_successfull_layers = {
            layer: TimeSeries(self.expected_num_days, dtype=int)
            for layer in self.graph.layer_ids
        }

    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        """Extend state initialisation with per-node disease-timeline arrays.

        After calling the parent :meth:`states_and_counts_init`:

        * Allocates ``infectious_time``, ``symptomatic_time``,
          ``rna_time``, and ``time_to_die`` arrays (all initialised to -1).
        * Resets ``self.testable``.
        * Sets ``self.need_check`` for S / S_s nodes.
        * Calls :meth:`update_plan` for all nodes to set initial plans.

        Args:
            ext_nodes (int, optional): Number of external nodes.
            ext_code (int, optional): State code for external nodes.
        """
        super().states_and_counts_init(ext_nodes, ext_code)


        self.infectious_time = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)
        self.symptomatic_time = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)
        self.rna_time = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)

        self.time_to_die = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)

        self.testable = np.zeros(
            self.num_nodes, dtype=bool)

        # need_check - state that needs regular checkup
        self.need_check = np.logical_or(
            self.memberships[STATES.S],
            self.memberships[STATES.S_s]
        )

        # todo: WTF why we need this?
        index = np.random.randint(37, size=10)
        self.time_to_go[index] = -1
        self.state_to_go[index] = -1

        # move all nodes to S and set move
        self.update_plan(np.ones(self.num_nodes, dtype=bool))

        
    def daily_update(self, nodes):
        """Perform daily infection checks and update plans for susceptible nodes.

        For susceptible (S / S_s) nodes:

        1. Optionally activates external-node edges (if external nodes present).
        2. Calls :func:`~models.prob_infection.prob_of_contact` to compute
           per-node exposure probabilities.
        3. Sets ``time_to_go=1`` and ``state_to_go=E`` for newly exposed nodes.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes that need a daily
                check (from ``self.need_check``).
        """
        # S, S_s
        target_nodes = np.logical_or(
            self._get_target_nodes(nodes, STATES.S),
            self._get_target_nodes(nodes, STATES.S_s)
        )

        # if we have external nodes
        if self.num_ext_nodes > 0:
            self.flip_coin_for_external_edges()

        # try infection (may rewrite S/Ss moves)
        P_infection = prob_of_contact(self,
                                      [STATES.S_s, STATES.S],
                                      [STATES.S,
                                          STATES.S_s,
                                          STATES.E,
                                          STATES.I_n,
                                          STATES.I_a,
                                       STATES.I_s
                                       ],
                                      [STATES.I_n, STATES.I_a,
                                          STATES.I_s, STATES.EXT],
                                      [STATES.I_n, STATES.I_a,
                                          STATES.I_s, STATES.E],
                                      self.beta, self.beta_in_family
                                      ).flatten()

        #    r = np.random.rand(target_nodes.sum())
        exposed = P_infection[target_nodes]
        # print(exposed, exposed.shape)
        # exit()

        exposed_mask = np.zeros(self.num_nodes, dtype=bool)
        exposed_mask[target_nodes] = exposed

        self.time_to_go[exposed_mask] = 1
        self.state_to_go[exposed_mask] = STATES.E


    def update_plan(self, nodes):
        """Generate new transition plans for nodes that just changed state.

        For each state, samples the appropriate duration(s) from the loaded
        duration distributions and sets ``self.time_to_go``,
        ``self.state_to_go``, and ``self.need_check`` accordingly.

        State-specific logic:

        * **S**: no scheduled transition; flagged for daily checks.
        * **E**: samples incubation duration; branches to I_n or I_a
          stochastically based on ``asymptomatic_rate``.
        * **I_n**: samples infectious + RNA-positivity durations; schedules
          J_n.
        * **I_a**: samples asymptomatic, infectious, and RNA durations;
          schedules I_s.
        * **I_s**: decides death outcome; schedules J_s or D.
        * **J_s** / **J_n**: schedules R (or D for dying nodes).
        * **R** / **D**: clears plan.

        Args:
            nodes (numpy.ndarray): Boolean bitmap of nodes whose plans should
                be regenerated.
        """
        # update plan
        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, STATES.S)
        # print("---")
        # print(target_nodes.shape)
        # print(self.time_to_go.shape)

        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = STATES.S
        self.need_check[target_nodes] = True

        # STATES.S_s:   "S_s",
        # target_nodes = self._get_target_nodes(nodes, STATES.S_s)
        # assert target_nodes.sum() == 0, "S_s
        # self.time_to_go[target_nodes] = 7
        # self.state_to_go[target_nodes] = STATES.S
        # self.need_check[target_nodes] = True

        # STATES.E:     "E",
        target_nodes = self._get_target_nodes(nodes, STATES.E)
        # print(f"target nodes {target_nodes.shape}")
        # print(f"self.time_to_go {self.time_to_go.shape}")

        # asymptotic or symptomatic branch?
        r = np.random.rand(target_nodes.sum())
        asymptomatic = r < self.asymptomatic_rate[target_nodes, 0]

        asymptomatic_nodes = target_nodes.copy()
        asymptomatic_nodes[target_nodes] = asymptomatic

        symptomatic_nodes = target_nodes.copy()
        symptomatic_nodes[target_nodes] = np.logical_not(asymptomatic)

        self.time_to_go[target_nodes] = self.rngd["E"].get(
            n=(target_nodes.sum(), 1))
        self.state_to_go[asymptomatic_nodes] = STATES.I_n
        self.state_to_go[symptomatic_nodes] = STATES.I_a
        self.need_check[target_nodes] = False

        # STATES.I_n:   "I_n",
        # need to generate I duratin and J durations
        target_nodes = self._get_target_nodes(nodes, STATES.I_n)
        n = target_nodes.sum()
        if n > 0:
            expected_i_time, expected_j_time = gen_tuple(
                2,
                (n, 1),
                self.rngd["I"],
                self.rngd["RNA"]
            )

            self.infectious_time[target_nodes] = expected_i_time
            self.rna_time[target_nodes] = expected_j_time
            self.time_to_go[target_nodes] = expected_i_time
            self.state_to_go[target_nodes] = STATES.J_n
            self.need_check[target_nodes] = False

        # STATES.I_a:   "I_a",
        target_nodes = self._get_target_nodes(nodes, STATES.I_a)

        # current infectious time (part of total infectious time)
        expected_a_time, expected_i_time, expected_j_time = gen_tuple(
            3,
            (target_nodes.sum(), 1),
            self.rngd["A"],
            self.rngd["I"],
            self.rngd["RNA"]
        )

        self.infectious_time[target_nodes] = expected_i_time
        assert np.all(expected_a_time < expected_i_time)
        self.symptomatic_time[target_nodes] = expected_i_time - expected_a_time
        self.rna_time[target_nodes] = expected_j_time

        self.time_to_go[target_nodes] = expected_a_time
        self.state_to_go[target_nodes] = STATES.I_s
        self.need_check[target_nodes] = False

        # STATES.I_s:   "I_s",
        target_nodes = self._get_target_nodes(nodes, STATES.I_s)

        # decide for testing (testing policy must be ON to be tested)
        n = target_nodes.sum()
        if n > 0:
            r = np.random.rand(n)
            self.testable[target_nodes] = r < self.test_rate[target_nodes, 0]

            self.will_die[target_nodes] = self.die_or_not_to_die(target_nodes)
        target_nodes_to_die = np.logical_and(
            target_nodes,
            self.will_die
        )
        self.time_to_die[target_nodes_to_die,
                         0] = self.get_time_to_die(target_nodes_to_die)

        nodes_to_die_now = np.zeros(len(target_nodes), dtype=bool)
        nodes_to_die_now[target_nodes_to_die] = self.time_to_die[target_nodes_to_die,
                                                                 0] <= self.symptomatic_time[target_nodes_to_die, 0]
        nodes_to_live_now = target_nodes
        nodes_to_live_now[nodes_to_die_now] = False

        # -> D
        self.time_to_go[nodes_to_die_now] = self.time_to_die[nodes_to_die_now]
        self.state_to_go[nodes_to_die_now] = STATES.D
        self.need_check[nodes_to_die_now] = False

        # -> J_s
        assert np.all(self.symptomatic_time[target_nodes] > 0)
        self.time_to_go[nodes_to_live_now] = self.symptomatic_time[nodes_to_live_now]
        self.state_to_go[nodes_to_live_now] = STATES.J_s
        self.need_check[nodes_to_live_now] = False
        self.time_to_die[nodes_to_live_now] = self.time_to_go[nodes_to_live_now]

        # STATES.J_s:   "J_s",
        target_nodes = self._get_target_nodes(nodes, STATES.J_s)

        nodes_to_die = np.logical_and(
            target_nodes,
            self.will_die
        )
        target_nodes[nodes_to_die] = False
        left_rna_positivity = self.rna_time[target_nodes] - \
            self.infectious_time[target_nodes]

        # -> D
        self.time_to_go[nodes_to_die] = self.time_to_die[nodes_to_die]
        self.state_to_go[nodes_to_die] = STATES.D
        self.need_check[nodes_to_die] = False

        # -> R
        assert np.all(self.symptomatic_time[target_nodes] > 0)
        self.time_to_go[target_nodes] = left_rna_positivity
        self.state_to_go[target_nodes] = STATES.R
        self.need_check[target_nodes] = False

        # STATES.J_n:   "J_n",
        target_nodes = self._get_target_nodes(nodes, STATES.J_n)

        left_rna_positivity = self.rna_time[target_nodes] - \
            self.infectious_time[target_nodes]

        self.time_to_go[target_nodes] = left_rna_positivity
        self.state_to_go[target_nodes] = STATES.R
        self.need_check[target_nodes] = False

        # STATES.R:   "R",
        target_nodes = self._get_target_nodes(nodes, STATES.R)
        self.testable[target_nodes] = False
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False

        # STATES.D:   "D",
        target_nodes = self._get_target_nodes(nodes, STATES.D)
        self.testable[target_nodes] = False
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False



    def run_iteration(self):
        """Perform one day of simulation with infection-time tracking.

        Before delegating to the parent :meth:`run_iteration`:

        * Resets per-layer successful-transmission counters for the current
          day.
        * Increments ``self.infect_time`` for currently infectious nodes.
        * When in debug mode, verifies that external-node state constraints
          are satisfied.
        """
        if self.num_ext_nodes > 0 and __debug__:
                # check that ext nodes are still ext nodes
                assert np.all(
                    self.memberships[STATES.EXT, self.nodes[:-self.num_ext_nodes], 0] == 0)
                assert np.all(
                    self.memberships[STATES.EXT, self.nodes[-self.num_ext_nodes:], 0] == 1)

                # check that ext nodes are not in quarantine
                if self.graph.is_quarantined is not None:
                    assert np.all(
                        self.graph.is_quarantined[self.nodes[-self.num_ext_nodes:]] == 0)


        for _, s in self.stat_successfull_layers.items():
            s[self.t] = 0

        infectious_nodes = (
            self.memberships[STATES.I_a] +
            self.memberships[STATES.I_s] +
            self.memberships[STATES.I_n]
        ).ravel()
        self.infect_time[infectious_nodes == 1] += 1

        super().run_iteration()

                
    def move_to_R(self, nodes):
        """Move a list of nodes to the Recovered state.

        Args:
            nodes (array-like of int): Node indices to recover.
        """
        target_nodes = np.zeros(self.num_nodes, dtype=bool)
        target_nodes[nodes] = True
        self.change_states(target_nodes, target_state=STATES.R)

    def move_target_nodes_to_R(self, target_nodes):
        """Move nodes (given as a boolean bitmap) to the Recovered state.

        Args:
            target_nodes (numpy.ndarray): Boolean bitmap of shape
                ``(num_nodes,)``; nodes where ``True`` are moved to R.
        """
        self.change_states(target_nodes, target_state=STATES.R)

    def move_target_nodes_to_S(self, target_nodes):
        """Move nodes (given as a boolean bitmap) to the Susceptible state.

        Args:
            target_nodes (numpy.ndarray): Boolean bitmap of shape
                ``(num_nodes,)``; nodes where ``True`` are moved to S.
        """
        self.change_states(target_nodes, target_state=STATES.S)

    def move_to_E(self, num):
        """Randomly expose *num* susceptible (S or S_s) nodes by moving them to E.

        Selects nodes uniformly at random from all currently susceptible nodes.
        If fewer susceptible nodes are available than requested, logs a warning
        and returns without action.

        Args:
            num (int): Number of nodes to expose.
        """

        # nodes_supply = [
        #    x
        #    for x in self.graph.nodes
        #    if (
        #            (self.graph.is_quarantined is None or not self.graph.is_quarantined[x])
        #            and
        #            self.memberships[STATES.R][x] != 1
        #    )
        # ]
        s_or_ss = np.logical_or(
            self.memberships[STATES.S],
            self.memberships[STATES.S_s]
        ).ravel()

        # s_or_ss = np.logical_and(
        #    s_or_ss,
        #    self.graph.nodes_age <= 20 # ucitele nenakazujeme
        # )

        nodes_supply = self.graph.nodes[s_or_ss]
        if len(nodes_supply) == 0:
            logging.warning("No nodes to infect.")
            return
        nodes = np.random.choice(nodes_supply, num, replace=False)

        target_nodes = np.zeros(self.num_nodes, dtype=bool)
        target_nodes[nodes] = True
        self.change_states(target_nodes, target_state=STATES.E)


    def df_source_infection(self):
        """Return a DataFrame of successful transmissions per contact layer per day.

        Returns:
            pandas.DataFrame: Indexed by day (0..t-1), with one column per
            graph layer named after the layer.
        """
        df = pd.DataFrame(index=range(0, self.t))
        for i in self.graph.layer_ids:
            df[self.graph.layer_name[i]] = self.stat_successfull_layers[i].asarray()[
                :self.t]
        return df


    def df_source_nodes(self):
        """Return a Series of successful infection counts per infectious node.

        Filters to nodes that are not in S, S_s, or E (i.e. nodes that were
        at some point infectious).

        Returns:
            pandas.Series: Successful infection counts indexed by node.
        """
        self.successfull_source_of_infection = self.successfull_source_of_infection[
            np.logical_and(
                np.logical_and(
                    self.current_state != STATES.S,
                    self.current_state != STATES.S_s),
                self.current_state != STATES.E).flatten()
        ]

        df = pd.Series(self.successfull_source_of_infection)
        return df


    def die_or_not_to_die(self, target_nodes):
        """Decide stochastically whether each target node will die from the disease.

        Uses age- and sex-stratified case-fatality rates scaled by ``self.mu``.

        Args:
            target_nodes (numpy.ndarray): Boolean bitmap of nodes entering
                the symptomatic state.

        Returns:
            numpy.ndarray: Boolean array of length ``target_nodes.sum()``;
            ``True`` where the node is destined to die.
        """
        n_target_nodes = target_nodes.sum()
        if n_target_nodes == 0:
            return np.array([], dtype=float)
        sex = self.graph.nodes_sex[target_nodes]
        age = self.graph.nodes_age[target_nodes].astype(int)
        probs = np.zeros(n_target_nodes, dtype=float)

        for sex_type in self.MALE, self.FEMALE:
            sel = sex == sex_type
            probs[sel] = self.mu * self.death_probs[sex_type][age[sel]]

        r = np.random.rand(n_target_nodes)
        return r < probs

    def get_time_to_die(self, target_nodes):
        """Sample the number of days until death for each dying node.

        Uses a mixed piecewise distribution:

        * If U < 0.571: ``X = ceil(10 * U / 0.571)``
        * Otherwise:  ``X = round(4 - ln(1 - U) / 0.13)``

        Args:
            target_nodes (numpy.ndarray): Boolean bitmap of nodes that have
                been determined to die (from :meth:`die_or_not_to_die`).

        Returns:
            numpy.ndarray: Integer array of length ``target_nodes.sum()``
            with days-until-death for each dying node.
        """
        # 1. Vygeneruj U z R(0,1)
        # 2. Pokud U < 0.571, pak X=ceil(U / 0.571)
        # 3. Jinak X= round(4+ln(1-U)/-0.13))

        n_target_nodes = target_nodes.sum()
        self.dtime_coef = 0.571
        random_u = np.random.rand(n_target_nodes)
        lower = random_u < self.dtime_coef
        higher = np.logical_not(lower)

        time_X = np.zeros(n_target_nodes, dtype=int)
        time_X[lower] = np.ceil(10*random_u[lower] / self.dtime_coef)
        time_X[higher] = np.round(4-np.log(1-random_u[higher])/0.13)

        assert np.all(self.time_to_die[target_nodes, 0] == -1)
        return time_X

    def get_dead(self):
        """Return summary statistics about deceased nodes.

        Returns:
            tuple: ``(total_dead, young, old1, old2)`` where:

            * ``total_dead`` (int) – total count of deceased nodes.
            * ``young`` (int) – deceased under 65.
            * ``old1`` (int) – deceased aged 65–79.
            * ``old2`` (int) – deceased aged 80+.
        """
        alld = self.state_counts[STATES.D][-1]
        dead_nodes = (self.memberships[STATES.D] == 1).flatten()
        ages = self.graph.nodes_age[dead_nodes]
        young = (ages < 65).sum()
        old1 = np.logical_and(ages >= 65, ages <= 79).sum()
        old2 = (ages >= 80).sum()
        return alld, young, old1, old2

    def flip_coin_for_external_edges(self):
        """Stochastically activate external-node edges for the current day.

        Switches all external-node edges on (resetting from the previous day),
        then switches off each edge independently with probability
        ``1 - self.ext_epi``.  This controls the likelihood that external
        (imported) cases interact with the main population on any given day.
        """
        ext_nodes = self.nodes[-self.num_ext_nodes:].ravel()
        ext_edges = self.graph.get_nodes_edges(list(ext_nodes))
        self.graph.switch_on_edges(ext_edges)  # recover from the last time

        r = np.random.rand(len(ext_edges))
        ext_edges_off = np.array(ext_edges)[r >= self.ext_epi]
        self.graph.switch_off_edges(list(ext_edges_off))
