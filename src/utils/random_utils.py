"""Random number generation utilities for the MAIS simulation.

This module provides helpers for reproducible random sampling, ordered-tuple
generation, and discrete duration sampling. It is used across the simulation
to draw stochastic values for disease progression durations and other
time-varying random quantities.
"""

import numpy as np
from numpy.random import Generator, SFC64
import logging


class RandomGenerator():
    """Seeded pseudo-random number generator wrapper using the SFC64 bit generator.

    Wraps NumPy's ``Generator`` backed by the fast ``SFC64`` bit generator so
    that independent, reproducible streams can be attached to different parts
    of the model.

    Note:
        Adding per-component generators is still in progress.

    Args:
        seed (int): Integer seed passed to ``SFC64`` for reproducibility.
    """

    # add own generators to individual parts of a model (in progress)

    def __init__(self, seed):
        self.rng = Generator(SFC64(seed))

    def rand(n):
        """Generate ``n`` uniformly distributed random floats in ``[0, 1)``.

        Args:
            n (int): Number of random values to generate.

        Returns:
            numpy.ndarray: Array of shape ``(n,)`` with values in ``[0, 1)``.
        """
        return rng.random(n)


def _random_from_probs(what, p, n=1):
    """Draw ``n`` random samples from ``what`` according to probabilities ``p``.

    Args:
        what (int or array-like): If an integer, samples are drawn from
            ``range(what)``. Otherwise samples are drawn from the provided
            sequence.
        p (array-like): Probability weights for each element of ``what``.
            Must sum to 1.
        n (int, optional): Number of samples to draw. Defaults to ``1``.

    Returns:
        numpy.ndarray: Array of ``n`` sampled values.
    """
    return np.random.choice(what, p=p, size=n)


def _check_sorted(value_list):
    """Check that corresponding elements across a list of arrays are strictly increasing.

    Given arrays ``[x1, x2, ..., xn]`` (each of the same shape), verifies
    element-wise that ``x1[i] < x2[i] < ... < xn[i]`` for every index ``i``.

    Args:
        value_list (list of numpy.ndarray): Ordered list of arrays to compare
            pairwise. All arrays must have the same shape.

    Returns:
        numpy.ndarray: Boolean array of the same shape as each input array.
        ``True`` at position ``i`` means all consecutive pairs satisfy the
        strict ordering at that position; ``False`` indicates a violation.
    """
    # print()
    # print(value_list)
    # print()
    partial_results_list = [
        np.atleast_1d(x < y)
        for x, y in zip(value_list[:-1], value_list[1:])
    ]
    #    print()

    #    print("partial_results_list", partial_results_list, len(partial_results_list))
    if len(partial_results_list) > 1:
        return np.logical_and(*partial_results_list)
    else:
        return partial_results_list[0]


def gen_tuple1(n, shape, *args):
    """Generate an ``n``-tuple of random values satisfying a strict ordering.

    Draws values ``(r_1, r_2, ..., r_n)`` such that ``r_1 < r_2 < ... < r_n``
    element-wise across a batch of size ``shape``. Any positions that violate
    the ordering are resampled repeatedly until all positions satisfy the
    constraint.

    Args:
        n (int): Number of elements in the tuple. Must equal ``len(args)``.
        shape (int or tuple): Shape of the batch to generate (passed as ``n``
            argument to each generator's ``get`` method).
        *args: Exactly ``n`` random duration generator objects, each exposing
            a ``get(n=...)`` method that returns a NumPy array of samples.

    Returns:
        list of numpy.ndarray: List of ``n`` arrays, each of shape ``shape``,
        satisfying ``result[0] < result[1] < ... < result[n-1]``
        element-wise.

    Example:
         >>> gen_tuple(3, rng1, rng2, rng3)
    """

    def _gen(s):
        result = []
        for i in range(n):
            result.append(args[i].get(n=s))
        return result

    assert len(args) == n

    result = _gen(shape)

    check = _check_sorted(result)
    while not np.all(check):
        loggin.info("gen_tuple: condition no satisfied, repairing")
        indices_to_fix = np.where(check == False)[0]
        new_values = _gen(indices_to_fix.shape[0])  # list of length n again
        # but with shorter items

        for i in range(n):
            result[i][indices_to_fix] = new_values[i].reshape(-1, 1)
        check = _check_sorted(result)

    return result


def gen_tuple2(n, shape, *args):
    """Generate an ``n``-tuple of random values satisfying a strict ordering (clipping variant).

    Draws values ``(r_1, r_2, ..., r_n)`` such that ``r_1 < r_2 < ... < r_n``
    element-wise across a batch of size ``shape``. Unlike :func:`gen_tuple1`,
    ordering is enforced by clipping each subsequent value to be at least
    ``previous + 1`` rather than by resampling.

    Args:
        n (int): Number of elements in the tuple. Must equal ``len(args)``.
        shape (int or tuple): Shape of the batch to generate (passed as ``n``
            argument to each generator's ``get`` method).
        *args: Exactly ``n`` random duration generator objects, each exposing
            a ``get(n=...)`` method that returns a NumPy array of samples.

    Returns:
        list of numpy.ndarray: List of ``n`` arrays, each of shape ``shape``,
        satisfying ``result[0] < result[1] < ... < result[n-1]``
        element-wise.

    Example:
         >>> gen_tuple(3, rng1, rng2, rng3)
    """
    result = []
    for i in range(n):
        values = args[i].get(n=shape)
        if i > 0:
            values = np.clip(values, result[i-1]+1, None)
        result.append(values)

    return result


def gen_tuple(n, shape, *args):
    """Generate an ``n``-tuple of strictly ordered random values.

    Delegates to :func:`gen_tuple2`. See that function for full documentation.

    Args:
        n (int): Number of elements in the tuple.
        shape (int or tuple): Shape of the batch to generate.
        *args: Exactly ``n`` random duration generator objects.

    Returns:
        list of numpy.ndarray: List of ``n`` strictly ordered arrays.
    """
    return gen_tuple2(n, shape, *args)


class RandomDuration():
    """Discrete random duration sampler driven by a full probability distribution.

    Intended for generating the time (in discrete steps, e.g. days) that an
    agent spends in a particular disease state. The distribution is specified
    as a probability mass function (PMF) over non-negative integer durations
    starting from zero.

    Args:
        probs (array-like): NumPy array of probabilities for durations
            ``0, 1, 2, ..., len(probs)-1``. Values must be non-negative and
            sum to 1.
        precompute (bool, optional): If ``True``, a large buffer of ``10^6``
            pre-drawn samples is generated at construction time. Currently the
            buffer is not stored, so this flag has no effect on subsequent
            ``get`` calls. Defaults to ``False``.
    """

    def __init__(self, probs, precompute=False):
        self.N = len(probs)
        self.probs = probs

        if precompute:
            buf = _random_from_probs(self.N, self.probs, 10**6)

    def get(self, n=1):
        """Draw ``n`` random duration values from the distribution.

        Args:
            n (int, optional): Number of samples to draw. Defaults to ``1``.

        Returns:
            numpy.ndarray: Array of ``n`` integer duration values drawn
            according to ``self.probs``.
        """
        values = _random_from_probs(self.N, self.probs, n)

        return values


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def uncumulate(l):
        res = [x - y
               for (x, y) in zip(l, [0]+l[:-1])
               ]
        s = sum(res)
        res[-1] += 1.0-s
        return res

    cdf_incubation = [0, 0.002079467, 0.045532967, 0.158206035, 0.303711753, 0.446245776, 0.569141375, 0.668484586,
                      0.746107988, 0.805692525, 0.851037774, 0.885435436, 0.911529759, 0.931365997, 0.946495014,
                      0.958080947, 0.966993762, 0.973882948, 0.979233968, 0.983410614, 0.986686454, 0.98926803,
                      0.991311965, 0.992937571, 0.994236158, 0.995277934, 0.996117131, 0.996795835, 0.997346849,
                      0.997795859, 0.998163058, 0.998464392, 0.998712499, 0.998917441, 0.999087255, 0.999228384,
                      0.999346016, 0.999444337, 0.999526742, 0.999595989, 0.999654327]
    p_incubation = uncumulate(cdf_incubation)

    values = []
    values2 = []
    durations = RandomDuration(p_incubation)
    pre_durations = RandomDuration(p_incubation, precompute=True)

    for _ in range(10000):

        values.extend(durations.get(100))
        values2.extend(durations.get(100))

    print(np.mean(values), np.median(values))
    print(np.mean(values2), np.median(values2))
    print(np.max(values), np.max(values2))
    max_value = max(values + values2)
    min_value = min(values + values2)

    fig, axs = plt.subplots(nrows=2, figsize=(10, 7))
    axs[0].hist(values, color="pink", label="onfly",
                bins=range(min_value, max_value+1))
    axs[0].hist(values2, color="blue", label="precomputed",
                bins=range(min_value, max_value+1))

    axs[1].hist(values2, color="blue", label="precomputed",
                bins=range(min_value, max_value+1))
    axs[1].hist(values, color="pink", label="onfly",
                bins=range(min_value, max_value+1))

    axs[0].set_xticks(range(min_value, max_value+1))
    axs[1].set_xticks(range(min_value, max_value+1))

    axs[0].legend()
    axs[1].legend()
    fig.suptitle("days in E")

    # Show plot
    plt.show()
