"""Base policy class for the MAIS epidemic simulation framework.

This module defines the abstract Policy base class that all concrete
intervention policies should subclass.
"""

import logging

class Policy:

    """Base Policy class.

    To implement a custom policy, derive your subclass and override
    ``first_day_setup``, ``run``, and optionally ``to_df``.

    Args:
        graph: The contact network graph object used by the simulation.
        model: The epidemic model instance that the policy acts upon.
    """

    def __init__(self, graph, model):
        """Initialise the policy with a graph and model.

        Args:
            graph: The contact network graph object.
            model: The epidemic model instance.
        """
        self.graph = graph
        self.model = model
        self.stopped = False
        self.first_day = True

    def first_day_setup(self):
        """Perform one-time setup on the first day the policy runs.

        Called automatically by ``run`` on the first invocation.
        Subclasses should override this method to perform any
        initialisation that requires the model to be running.
        """
        pass

    def run(self):
        """Execute one time-step of the policy.

        On the first call ``first_day_setup`` is invoked before the
        main logic.  Subclasses should call ``super().run()`` to
        preserve this behaviour.
        """
        if self.first_day:
            self.first_day_setup()
            self.first_day = False

        logging.info(
            f"This is the {self.__class__.__name__} policy run.  {'(STOPPED)' if self.stopped else ''}")


    def to_df(self):
        """Return a DataFrame with policy-related statistics.

        The returned DataFrame must contain a column ``T`` with the
        corresponding ``self.model.T`` date values.  An empty
        DataFrame or ``None`` is acceptable when no statistics have
        been collected.

        Returns:
            pandas.DataFrame or None: DataFrame of policy statistics,
            or ``None`` if not implemented.
        """
        logging.warning("NOT IMPLEMENTED YET (to_df)")
        return None

