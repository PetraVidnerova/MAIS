"""Deposit (Depo) utility for tracking nodes held for a fixed duration.

This module provides the :class:`Depo` class used by policy objects to
place nodes into quarantine or isolation for a configurable number of
time-steps and to release them automatically when their sentence
expires.
"""

import numpy as np
import logging

class Depo:

    """Deposit object for holding nodes for a fixed number of time-steps.

    Used in policies to store nodes (e.g. quarantined individuals) for a
    given period of time.  Internally maintains a counter array; each
    entry holds the remaining number of time-steps before that node is
    released.  A value of zero means the node is not currently held.

    Args:
        size (int): Total number of nodes in the simulation (length of
            the internal counter array).
    """

    def __init__(self, size):
        """Initialise the deposit with a counter array of zeros.

        Args:
            size (int): Number of nodes (length of the counter array).
        """
        self.depo = np.zeros(size, dtype="uint8")

    @property
    def num_of_prisoners(self):
        """Return the number of nodes currently held in the deposit.

        Returns:
            int: Count of nodes whose remaining time is greater than zero.
        """
        return (self.depo > 0).sum()

    def lock_up(self, nodes, duration=14, check_duplicate=False):
        """Place nodes into the deposit for a given duration.

        Args:
            nodes (list or numpy.ndarray): Node indices to lock up.
            duration (int or numpy.ndarray): Number of time-steps each
                node should remain in the deposit.  May be a scalar
                applied to all nodes or an array of per-node values.
                Defaults to 14.
            check_duplicate (bool): If ``True``, nodes that are already
                in the deposit are silently ignored (their remaining
                time is not updated).  If ``False`` (default), an
                assertion error is raised if any node is already held.

        Raises:
            AssertionError: If ``nodes`` is not a list or
                ``numpy.ndarray``, or if ``check_duplicate`` is
                ``False`` and any node is already in the deposit.
        """
        assert isinstance(nodes, np.ndarray) or isinstance(nodes, list), f"real type {type(nodes)}"
        if len(nodes) > 0:
            assert check_duplicate or np.all(self.depo[nodes] == 0)

            if check_duplicate:
                nodes = np.array(nodes)
                zero_nodes = nodes[(self.depo[nodes] == 0).nonzero()[0]]
                self.depo[zero_nodes] = duration
            else:
                self.depo[nodes] = duration

    def filter_locked(self, candidates):
        """Return the subset of candidates that are not yet in the deposit.

        Args:
            candidates (numpy.ndarray): Array of node indices to filter.

        Returns:
            numpy.ndarray: Subset of ``candidates`` whose deposit
            counter is currently zero (i.e. not locked up).
        """
        if len(candidates) > 0:
            return candidates[self.depo[candidates]==0]
        else:
            return candidates
    

    def filter_locked_bitmap(self, candidates):
        """Return a boolean mask indicating which candidates are not in the deposit.

        Args:
            candidates (numpy.ndarray): Array of node indices to check.

        Returns:
            numpy.ndarray: Boolean array of the same length as
            ``candidates``.  Entry is ``True`` if the corresponding
            candidate is **not** currently in the deposit, ``False``
            if it is already locked up.
        """
        return self.depo[candidates] == 0

    def tick_and_get_released(self):
        """Advance the deposit by one time-step and return newly released nodes.

        Decrements all non-zero counters by one.  Nodes whose counter
        reaches zero are considered released.

        Returns:
            numpy.ndarray: Array of node indices that were released
            (i.e. had a counter of exactly 1 before the decrement).
        """
        released = np.nonzero(self.depo == 1)[0]
        self.depo[self.depo >= 1] -= 1
        return released

    def is_locked(self, node_id):
        """Check whether a single node is currently held in the deposit.

        Args:
            node_id (int): Index of the node to check.

        Returns:
            bool: ``True`` if the node's remaining time is greater than
            zero, ``False`` otherwise.
        """
        return self.depo[node_id] > 0
