"""Highly configurable custom policy for the MAIS epidemic simulation.

This module provides :class:`CustomPolicy`, the primary policy used in
production simulations.  It orchestrates:

* Layer-weight changes read from a scenario calendar file.
* Model parameter changes (beta, theta, test rate).
* Face-mask effectiveness updates.
* Superspreader event toggling.
* Forced infections on a specified day.
* Daily background import of exposed individuals.
* Dynamic start/stop of sub-policies loaded by name.
"""

import pandas as pd
import numpy as np
import json
from functools import partial, reduce
import logging

from policies.policy import Policy
from utils.policy_utils import load_scenario_dict


def _load_dictionary(filename: str):
    """Load a time-indexed dictionary from a JSON or CSV file.

    JSON files are expected to be objects with string keys (simulation
    day numbers) and arbitrary values.  CSV files must have a ``T``
    column and a second column whose values are used.

    Args:
        filename (str or None): Path to the file, or ``None``.

    Returns:
        dict or None: Mapping from ``int`` day to the corresponding
        value, or ``None`` if ``filename`` is ``None``.
    """
    if filename is None:
        return None
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            return {
                int(key): value
                for key, value in json.load(f).items()
            }
    else:
        df = pd.read_csv(filename)
        return dict(zip(df["T"], df.iloc[:, 1]))


class CustomPolicy(Policy):

    """Highly configurable orchestration policy for MAIS simulations.

    Controls layer weights, model parameters (beta, theta, test rate),
    face-mask effectiveness, superspreader events, forced infections,
    background import of exposed individuals, and the lifecycle of
    dynamically loaded sub-policies.

    All time-indexed inputs are read from external files (JSON or CSV)
    keyed on integer simulation-day numbers.

    Args:
        graph: The contact network graph object.
        model: The epidemic model instance.
        layer_changes_filename (str, optional): Path to a scenario
            calendar file for layer-weight updates.
        param_changes_filename (str, optional): Temporarily disabled.
            Path to a JSON file of model-parameter changes.
        policy_calendar_filename (str, optional): Path to a JSON file
            mapping simulation days to ``[action, policy_string]``
            pairs for starting/stopping sub-policies.
        beta_factor_filename (str, optional): Path to a
            JSON/CSV file with per-day beta-factor multipliers.
        face_masks_filename (str, optional): Path to a JSON/CSV file
            with per-day face-mask compliance values.
        theta_filename (str, optional): Path to a JSON/CSV file with
            per-day ``theta_Is`` multipliers.
        test_rate_filename (str, optional): Path to a JSON/CSV file
            with per-day test-rate multipliers.
        superspreader_date (int or str, optional): Simulation day on
            which the superspreader layer is activated (one day only).
        superspreader_layer (int or str, optional): Index of the
            superspreader layer (default 31).
        force_infect (int or str, optional): Simulation day on which
            one node on ``force_infect_layer`` is forcibly infected.
        force_infect_layer (int or str, optional): Layer index used to
            select the node for forced infection.
        init_filename (str, optional): Path to a JSON file mapping days
            to the number of nodes moved to state E on that day.
        reduction_coef1 (float or str): Required.  Primary beta
            reduction coefficient (face-mask effect on non-family
            contacts).
        reduction_coef2 (float or str): Required.  Secondary beta
            reduction coefficient (face-mask spillover into family).
        new_beta (str, optional): ``"Yes"`` to use the updated beta
            calculation combining face-mask compliance and beta factors.
        daily_import (float or str, optional): Daily probability of
            importing one exposed individual (value in [0, 1]).
        **kwargs: Sub-policy keyword arguments.  Pass
            ``sub_policies`` (list of names) together with
            ``<name>_filename``, ``<name>_name``, and optionally
            ``<name>_config`` for each sub-policy.

    Raises:
        ValueError: If ``param_changes_filename`` is provided (temporarily
            disabled), if an unknown action is found in the policy
            calendar, if ``new_beta=Yes`` but the required calendars are
            absent or mismatched, or if ``reduction_coef1``/
            ``reduction_coef2`` are missing.
    """

    def __init__(self,
                 graph,
                 model,
                 layer_changes_filename=None,
                 param_changes_filename=None,
                 policy_calendar_filename=None,
                 beta_factor_filename=None,
                 face_masks_filename=None,
                 theta_filename=None,
                 test_rate_filename=None,
                 superspreader_date=None,
                 superspreader_layer=None,
                 force_infect=None,
                 force_infect_layer=None,
                 init_filename=None,
                 reduction_coef1=None,
                 reduction_coef2=None,
                 new_beta=None,
                 daily_import=None,
                 **kwargs):
        """Initialise the custom policy from file-based calendars and keyword arguments.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
            layer_changes_filename (str, optional): Layer-weight
                scenario calendar file.
            param_changes_filename (str, optional): Temporarily
                disabled model-parameter changes file.
            policy_calendar_filename (str, optional): Sub-policy
                lifecycle calendar file.
            beta_factor_filename (str, optional): Beta-factor calendar
                file.
            face_masks_filename (str, optional): Face-mask compliance
                calendar file.
            theta_filename (str, optional): ``theta_Is`` multiplier
                calendar file.
            test_rate_filename (str, optional): Test-rate multiplier
                calendar file.
            superspreader_date (int or str, optional): Day to activate
                superspreader layer.
            superspreader_layer (int or str, optional): Superspreader
                layer index.
            force_infect (int or str, optional): Day for forced
                infection.
            force_infect_layer (int or str, optional): Layer for
                selecting the force-infected node.
            init_filename (str, optional): Initial exposure calendar
                file.
            reduction_coef1 (float or str): Primary beta reduction
                coefficient (required).
            reduction_coef2 (float or str): Secondary beta reduction
                coefficient (required).
            new_beta (str, optional): ``"Yes"`` to use new beta
                calculation.
            daily_import (float or str, optional): Daily import
                probability.
            **kwargs: Sub-policy configuration keyword arguments.
        """
        super().__init__(graph, model)

        if layer_changes_filename is not None:
            self.layer_changes_calendar = load_scenario_dict(
                layer_changes_filename)
        else:
            self.layer_changes_calendar = None

        if policy_calendar_filename is not None:
            with open(policy_calendar_filename, "r") as f:
                self.policy_calendar = {
                    int(k): v
                    for k, v in json.load(f).items()
                }
        else:
            self.policy_calendar = None

        if param_changes_filename is not None:
            raise ValueError("Temporarily disabled. Sry.")
            with open(param_changes_filename, "r") as f:
                self.param_changes_calendar = json.load(f)
        else:
            self.param_changes_calendar = None

        self.face_masks_calendar = _load_dictionary(face_masks_filename)
        if self.face_masks_calendar is None:
            logging.warning("DBG: NO MASKS ")

        self.beta_factor_calendar = _load_dictionary(beta_factor_filename)

        self.theta_calendar = _load_dictionary(theta_filename)
        if self.theta_calendar is None:
            logging.warning("DBG: Theta calendar")

        self.test_rate_calendar = _load_dictionary(test_rate_filename)

        self.policies = {}

        if superspreader_date is not None:
            self.superspreader_date = int(superspreader_date)
        else:
            self.superspreader_date = None

        if superspreader_layer is not None:
            self.superspreader_layer = int(superspreader_layer)
        else:
            self.superspreader_layer = 31

        if force_infect is not None:
            self.force_infect = int(force_infect)
            assert self.force_infect
        else:
            self.force_infect = None

        if force_infect_layer is not None:
            self.force_infect_layer = int(force_infect_layer)
        else:
            self.force_infect_layer = None

        if init_filename is not None:
            with open(init_filename, "r") as f:
                self.init_calendar = {
                    int(k): v
                    for k, v in json.load(f).items()
                }
        else:
            self.init_calendar = None

        if reduction_coef1 is not None:
            self.reduction_coef1 = float(reduction_coef1)
        else:
            self.reduction_coef1 = 0.9
            raise ValueError("Missing coef1")

        if reduction_coef2 is not None:
            self.reduction_coef2 = float(reduction_coef2)
        else:
            self.reduction_coef2 = 0.2
            raise ValueError("Missing coef2")

        self.mutation_coef = 1.0

        self.new_beta = False if new_beta is None else new_beta == "Yes"
        if self.new_beta:
            logging.debug("Using new beta")
            assert self.beta_factor_calendar is not None
            assert self.face_masks_calendar is not None
            assert self.beta_factor_calendar.keys() == self.face_masks_calendar.keys()

        self.nodes_infected = None

        if daily_import is not None:
            self.daily_import = float(daily_import)
        else:
            self.daily_import = daily_import

        if "sub_policies" in kwargs:
            if type(kwargs["sub_policies"]) is list:
                subpolicies = kwargs["sub_policies"]
            else:
                subpolicies = [kwargs["sub_policies"]]
            for sub_policy_name in subpolicies:
                filename = kwargs[f"{sub_policy_name}_filename"]
                class_name = kwargs[f"{sub_policy_name}_name"]
                config = kwargs.get(f"{sub_policy_name}_config", None)

                policy = f"{filename}:{class_name}" if config is None else f"{filename}:{class_name}:{config}"
                self.policies[policy] = self.create_policy(
                    filename, class_name, config)

    def create_policy(self, filename, object_name, config_file=None):
        """Dynamically import and instantiate a policy class by module and name.

        Args:
            filename (str): Module name within the ``policies`` package
                (e.g. ``"contact_tracing"``).
            object_name (str): Class name within that module.
            config_file (str, optional): Path to a configuration file
                to pass to the policy constructor.

        Returns:
            Policy: An instantiated policy object.
        """
        PolicyClass = getattr(
            __import__(
                "policies."+filename,
                globals(), locals(),
                [object_name],
                0
            ),
            object_name
        )
        if config_file:
            return PolicyClass(self.graph, self.model, config_file=config_file)
        else:
            return PolicyClass(self.graph, self.model)

    def update_layers(self, coefs):
        """Update graph layer weights to the provided coefficient values.

        Args:
            coefs: Iterable of new layer-weight values passed directly
                to ``graph.set_layer_weights``.
        """
        self.graph.set_layer_weights(coefs)

    def switch_on_superspread(self):
        """Activate the superspreader layer by setting its weight to 1.0."""
        logging.info("DBG Superspreader ON")
        self.graph.layer_weights[self.superspreader_layer] = 1.0

    def switch_off_superspread(self):
        """Deactivate the superspreader layer by setting its weight to 0.0."""
        logging.info("DBG Superspreader OFF")
        self.graph.layer_weights[self.superspreader_layer] = 0.0

    def update_beta(self, masks):
        """Update model beta values based on face-mask compliance (legacy method).

        Scales non-family beta by ``(1 - reduction_coef1 * masks)`` and
        family beta by a secondary factor derived from the non-family
        reduction.

        Args:
            masks (float): Current face-mask compliance level in [0, 1].
        """
        orig_beta = self.model.init_kwargs["beta"]
        orig_beta_A = self.model.init_kwargs["beta_A"]

        reduction = (1 - self.reduction_coef1 * masks)

        new_value = orig_beta * reduction
        new_value_A = orig_beta_A * reduction

        logging.debug(f"{self.model.T} DBG BETA {new_value}")

        self.model.beta.fill(new_value)
        self.model.beta_A.fill(new_value_A)

        orig_beta_in_family = self.model.init_kwargs["beta_in_family"]
        orig_beta_A_in_family = self.model.init_kwargs["beta_A_in_family"]

        reduction = 1 - self.reduction_coef2 * (1-reduction)

        new_value = orig_beta_in_family * reduction
        new_value_A = orig_beta_A_in_family * reduction
        self.model.beta_in_family.fill(new_value)
        self.model.beta_A_in_family.fill(new_value_A)

        logging.debug(f"DBG beta: {self.model.beta[0][0]} {self.model.beta_in_family[0][0]}")

    def update_beta2(self, masks, beta_factors=None):
        """Update model beta values using both face-mask compliance and beta factors.

        Family beta is reduced by ``(1 - reduction_coef1 * beta_factors)``.
        Non-family beta is then derived as
        ``(1 - reduction_coef2 * masks) * family_beta``.

        Args:
            masks (float): Current face-mask compliance in [0, 1].
            beta_factors (float): Additional beta scaling factor.

        Raises:
            AssertionError: If ``beta_factors`` is ``None``.
        """
        assert beta_factors is not None

        orig_beta_in_family = self.model.init_kwargs["beta_in_family"]
        orig_beta_A_in_family = self.model.init_kwargs["beta_A_in_family"]

        reduction = (1 - self.reduction_coef1*beta_factors)

        new_value = self.mutation_coef * orig_beta_in_family * reduction
        new_value_A = self.mutation_coef * orig_beta_A_in_family * reduction
        self.model.beta_in_family.fill(new_value)
        self.model.beta_A_in_family.fill(new_value_A)

        #orig_beta= self.model.init_kwargs["beta"]
        #orig_beta_A = self.model.init_kwargs["beta_A"]

        # assumes betas are uniform
        new_beta = (1-self.reduction_coef2*masks) * \
            self.model.beta_in_family[0][0]
        new_beta_A = (1-self.reduction_coef2*masks) * \
            self.model.beta_A_in_family[0][0]

        logging.debug(f"{self.model.T} DBG BETA {new_value}")

        self.model.beta.fill(new_beta)
        self.model.beta_A.fill(new_beta_A)

        logging.debug(f"DBG beta: {self.model.beta[0][0]} {self.model.beta_A[0][0]}")

    def beta_increase(self):
        """Increase all beta values by a factor of 1.5 (mutation simulation).

        Sets ``mutation_coef`` to 1.5 and scales ``beta``, ``beta_A``,
        ``beta_A_in_family``, and ``beta_in_family`` accordingly.
        """
        self.mutation_coef = 1.5

        orig_value = self.model.beta[0][0]
        self.model.beta.fill(self.mutation_coef*orig_value)

        orig_value = self.model.beta_A[0][0]
        self.model.beta_A.fill(self.mutation_coef*orig_value)

        orig_value = self.model.beta_A_in_family[0][0]
        self.model.beta_A_in_family.fill(self.mutation_coef*orig_value)

        orig_value = self.model.beta_in_family[0][0]
        self.model.beta_in_family.fill(self.mutation_coef*orig_value)

    def update_test_rate(self, coef):
        """Scale the testing rate (theta_Is) by a given coefficient.

        Args:
            coef (float): Multiplier applied to the original test rate
                from ``model.init_kwargs``.
        """
        orig_test_rate = self.model.init_kwargs["test_rate"]
        new_value = coef * orig_test_rate
        #self.model.test_rate = new_value
        self.model.theta_Is.fill(new_value)

    def update_theta(self, coef):
        """Scale the symptomatic testing probability (theta_Is) by a coefficient.

        Args:
            coef (float): Multiplier applied to the original
                ``theta_Is`` value from ``model.init_kwargs``.
        """
        orig_theta = self.model.init_kwargs["theta_Is"]
        new_value = orig_theta * coef
        self.model.theta_Is.fill(new_value)
        # if isinstance(new_value, (list)):
        #     np_new_value = np.array(new_value).reshape(
        #         (self.model.num_nodes, 1))
        # else:
        #     np_new_value = np.full(
        #         fill_value=new_value, shape=(self.model.num_nodes, 1))
        # setattr(self.model, "theta_Is", np_new_value)
        logging.debug(f"DBG theta: {self.model.theta_Is[0][0]}")

    def run(self):
        """Execute one time-step of the custom policy.

        Processes all registered calendars in order:
        initial exposures, daily import, policy-calendar events,
        parameter changes, layer updates, forced infections,
        superspreader toggling, face-mask updates, theta updates,
        test-rate updates, and finally runs all active sub-policies.
        """
        if False and self.graph.is_quarantined is not None:
            # dbg check
            all_deposited = np.zeros(self.graph.number_of_nodes)
            for p in self.policies.values():
                all_deposited = all_deposited + p.depo.depo
                if isinstance(p, EvaQuarantinePolicy) or isinstance(p, ContactTracingPolicy):
                    all_deposited += p.waiting_room_second_test.depo
            assert np.sum(all_deposited > 0) == np.sum(self.graph.is_quarantined > 0),  f"{all_deposited.nonzero()[0]} \n {self.graph.is_quarantined.nonzero()[0]}"

            # print(all_deposited.nonzero()[0],
            #     self.graph.is_quarantined.nonzero()[0])
            assert np.all(
                all_deposited.nonzero()[0] == self.graph.is_quarantined.nonzero()[0]), f"{all_deposited.nonzero()[0]} \n {self.graph.is_quarantined.nonzero()[0]}"

        logging.info(f"CustomPolicy {int(self.model.T)}")
        today = int(self.model.T)

        if self.init_calendar is not None and today in self.init_calendar:
            num = self.init_calendar[today]
            self.model.move_to_E(num)

        if self.daily_import is not None:
            assert self.daily_import <= 1.0, "not implemented yet"
            if np.random.rand() < self.daily_import:
                self.model.move_to_E(1)

        if self.policy_calendar is not None and today in self.policy_calendar:
            logging.info("changing quarantine policy")
            # change the quaratine policy function
            for action, policy in self.policy_calendar[today]:
                if action == "start":
                    vals = policy.strip().split(":")
                    filename, object_name = vals[0], vals[1]
                    config_file = None if len(vals) == 2 else vals[2]
                    PolicyClass = getattr(__import__(filename), object_name)
                    if config_file is None:
                        self.policies[policy] = PolicyClass(
                            self.graph, self.model)
                    else:
                        self.policies[policy] = PolicyClass(
                            self.graph, self.model, config_file=config_file)
                    #params = [ float(param) for param in vals[2:] ]
                    #self.policies[policy] = PolicyClass(self.graph, self.model, *params)
                elif action == "stop":
                    self.policies[policy].stop()
                else:
                    raise ValueError(f"Unknown action {action}")

        if self.param_changes_calendar is not None and today in self.param_changes_calendar:
            for action, param, new_value in self.param_changes_calendar[today]:
                if action == "set":
                    if isinstance(new_value, (list)):
                        np_new_value = np.array(new_value).reshape(
                            (self.model.num_nodes, 1))
                    else:
                        np_new_value = np.full(
                            fill_value=new_value, shape=(self.model.num_nodes, 1))
                    setattr(self.model, param, np_new_value)
                elif action == "*":
                    attr = getattr(self.model, param)
                    if type(new_value) == str:
                        new_value = getattr(self.model, new_value)
                    setattr(self.model, param, attr * new_value)
                else:
                    raise ValueError("Unknown value")

        if self.layer_changes_calendar is not None and today in self.layer_changes_calendar:
            logging.debug(f"{today} updating layers")
            self.update_layers(self.layer_changes_calendar[today])

        if self.force_infect is not None and self.model.T == self.force_infect:
            # number_to_infect = 5 if self.force_infect_layer in (33,34,35) else 10
            number_to_infect = 1
            nodes_on_layer = self.graph.get_nodes(self.force_infect_layer)
            nodes_to_infect = np.random.choice(
                nodes_on_layer, number_to_infect, replace=False)
            self.model.force_infect(nodes_to_infect)
            self.nodes_infected = nodes_to_infect
 #           self.model.test_rate[self.nodes_infected] = 1.0
 #           self.model.theta_Is[self.nodes_infected] =  0.0
 #           self.model.testable[self.nodes_infected] = True

#        if self.nodes_infected is not None:
#            if self.model.t == self.force_infect + 6:
#                self.model.theta_Is[self.nodes_infected] =  1.0
#                self.nodes_infected = None

        if self.superspreader_date is not None:
            if self.model.T == self.superspreader_date:
                self.switch_on_superspread()
            elif self.model.T - 1 == self.superspreader_date:
                self.switch_off_superspread()

        if self.face_masks_calendar is not None and today in self.face_masks_calendar:
            logging.debug(f"DBG face masks update", self.face_masks_calendar)
            if self.new_beta:
                self.update_beta2(
                    self.face_masks_calendar[today], self.beta_factor_calendar[today])
            else:
                raise ValueError("Temporarily disabled.")

        # if today == 335:
        #     self.beta_increase()

        if self.theta_calendar is not None and today in self.theta_calendar:
            logging.debug(f"DBG theta update")
            self.update_theta(self.theta_calendar[today])

        if self.test_rate_calendar is not None and today in self.test_rate_calendar:
            logging.debug(f"DBG test rate update")
            self.update_test_rate(self.test_rate_calendar[today])

        # perform registred policies
        for name, policy in self.policies.items():
            logging.info(f"run policy { name}")
            policy.run()

    def to_df(self):
        """Merge and return DataFrames from all active sub-policies.

        Returns:
            pandas.DataFrame or None: Outer-merged DataFrame indexed by
            ``T`` combining statistics from all sub-policies, or
            ``None`` if there are no sub-policies or none produce data.
        """
        if not self.policies:
            return None

        dfs = [
            p.to_df()
            for p in self.policies.values()
        ]
        dfs = [d for d in dfs if d is not None]
        if not dfs:
            return None
        my_merge = partial(pd.merge, on="T", how="outer")

        return reduce(my_merge, dfs)
