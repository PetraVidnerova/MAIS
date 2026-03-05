"""Time-series and history container classes for the MAIS simulation.

This module provides a hierarchy of array-backed containers that record
simulation state over time. Containers grow automatically when new entries
are appended beyond their initial capacity, and they can be trimmed to the
actual simulation length at the end of a run.
"""

import numpy as np


class BaseSeries():

    """Base class for all time-series containers in the MAIS simulation.

    A ``BaseSeries`` stores values produced sequentially over simulation
    time—one item per iteration. Concrete subclasses back the storage with a
    NumPy array and override :meth:`bloat` to extend that array when needed.

    The container supports standard index-based read/write access. On a write
    to the index immediately past the current end of the array, the array is
    automatically extended by 100 elements before the write proceeds.
    """

    def __init__(self):
        self.values = None

    def __getitem__(self, idx):
        """Return the value at ``idx``.

        Args:
            idx (int or slice): Index or slice into the underlying array.

        Returns:
            The element (or sub-array) at the given position.
        """
        return self.values[idx]

    def __setitem__(self, idx, data):
        """Set the value at ``idx``, auto-extending the array if necessary.

        If ``idx`` equals the current length of the underlying array, the
        array is grown by 100 elements via :meth:`bloat` before the assignment
        is made.

        Args:
            idx (int): Target index. Must be <= ``len(self.values)``.
            data: Value to store at position ``idx``.

        Raises:
            IndexError: If ``idx`` is out of range and not equal to
                ``len(self.values)``.
        """
        try:
            self.values[idx] = data
        except IndexError as e:
            if idx == len(self.values):
                self.bloat(100)
                self.values[idx] = data
            else:
                print(len(self.values))
                print(idx)
                raise e

    def save(self, filename):
        """Save the underlying array to a NumPy ``.npy`` file.

        Args:
            filename (str): Destination file path (passed to
                ``numpy.save``).
        """
        np.save(self.values, filename)

    def __len__(self):
        """Return the number of elements in the series.

        Returns:
            int: Length of the underlying array.
        """
        return len(self.values)

    def len(self):
        """Return the number of elements in the series.

        Returns:
            int: Length of the underlying array.
        """
        return len(self.values)

    def asarray(self):
        """Return the underlying NumPy array directly.

        Returns:
            numpy.ndarray: The internal storage array.
        """
        return self.values

    def bloat(self, len):
        """Extend the internal storage by ``len`` elements.

        Subclasses must override this method with an implementation that
        appends ``len`` default-initialised elements to ``self.values``.

        Args:
            len (int): Number of elements to add.
        """
        pass


class TimeSeries(BaseSeries):

    """One-dimensional time series backed by a NumPy array of a given dtype.

    Stores one scalar value per simulation time-step. The array is
    pre-allocated with zeros and is extended automatically if needed.

    Args:
        len (int): Initial capacity (number of time-steps to pre-allocate).
        dtype (type, optional): NumPy dtype for the underlying array.
            Defaults to ``float``.
    """

    def __init__(self, len, dtype=float):
        super().__init__()
        self.type = dtype
        self.values = np.zeros(len, dtype)

    def bloat(self, len):
        """Extend the series by ``len`` zero-initialised elements.

        Args:
            len (int): Number of additional zero elements to append.
        """
        self.values = np.pad(
            self.values,
            [(0, len)],
            mode='constant', constant_values=0)

    def finalize(self, tidx):
        """Trim the series to the actual simulation length.

        Removes trailing zero-padding so that the array ends at the last
        recorded time-step.

        Args:
            tidx (int): Zero-based index of the last recorded time-step.
                Elements beyond this index are discarded.
        """
        self.values = self.values[:tidx+1]

    def get_values(self):
        """Return the underlying NumPy array.

        Returns:
            numpy.ndarray: The internal 1-D storage array.
        """
        return self.values


class TransitionHistory(BaseSeries):

    """Two-dimensional history table recording state-transition events over time.

    Each row corresponds to one recorded event (or time-step), and the fixed
    number of columns (``width``) encodes the details of each transition
    (e.g., node ID, source state, target state). The row count grows
    automatically; the column count is fixed at construction.

    Args:
        len (int): Initial row capacity (number of events to pre-allocate).
        dtype (type, optional): NumPy dtype for the underlying 2-D array.
            Defaults to ``int``.
        width (int, optional): Number of columns per row. Defaults to ``3``.
    """

    def __init__(self, len, dtype=int, width=3):
        super().__init__()
        self.values = np.zeros((len, width), dtype=dtype)
        self.width = width
        self.dtype = dtype

    def bloat(self, len):
        """Extend the table by ``len`` zero-initialised rows.

        Args:
            len (int): Number of additional rows to append.
        """
        new_space = np.zeros((len, self.width), dtype=self.dtype)
        self.values = np.vstack([self.values, new_space])

    def finalize(self, tidx):
        """Trim the table to the actual number of recorded events.

        Removes trailing zero-padded rows so that the table contains only
        the rows up to and including ``tidx``.

        Args:
            tidx (int): Zero-based index of the last recorded event.
                Rows beyond this index are discarded.
        """
        self.values = self.values[:tidx+1, :]


class ShortListSeries():
    """Fixed-capacity FIFO list that discards the oldest element when full.

    Behaves like a sliding window: once the list reaches ``length`` items,
    appending a new value automatically removes the oldest one (index 0).
    Useful for maintaining a rolling window of recent simulation metrics.

    Args:
        length (int): Maximum number of elements to retain.
    """

    def __init__(self, length):
        self.values = []
        self.length = length

    def append(self, member):
        """Append ``member`` and evict the oldest element if at capacity.

        Args:
            member: The value to add to the end of the list.
        """
        self.values.append(member)
        if len(self.values) > self.length:
            self.values.pop(0)

    def __getitem__(self, idx):
        """Return the element at ``idx``.

        Args:
            idx (int or slice): Index into the internal list.

        Returns:
            The element (or sub-list) at the given position.
        """
        return self.values[idx]

    def __len__(self):
        """Return the current number of stored elements.

        Returns:
            int: Number of elements currently in the list.
        """
        return len(self.values)
