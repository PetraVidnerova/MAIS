"""Global configuration parameters and utilities for the MAIS simulation.

This module provides module-level constants that act as global toggles and
settings used throughout the simulation. It also exposes a lightweight
monitoring helper that emits structured log messages for a designated node.

These globals should only be mutated when truly necessary; prefer passing
configuration explicitly through function arguments or config objects.
"""

import logging
from logging import StreamHandler
import sys


# this should be used only if really needed
# global config parameters, usually constants


MONITOR_NODE = None
SAVE_NODES = False
SAVE_DURATIONS = False


def monitor(t, msg):
    """Emit a monitoring log message for the globally watched node.

    Logs an INFO-level message that identifies the current simulation day,
    the monitored node (``MONITOR_NODE``), and an arbitrary status message.
    The function is a no-op in terms of return value; its purpose is purely
    for diagnostic logging during a simulation run.

    Args:
        t (int): The current simulation day (time-step index).
        msg (str): A descriptive message about the node's current status or
            event to be recorded.
    """
    logging.info(f"(Day {t}) NODE-MONITOR: Node {MONITOR_NODE} {msg}")
