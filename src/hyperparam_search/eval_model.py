"""Evaluation metrics for comparing model output to observed data.

This module provides functions to extract model predictions and compute
common regression-quality metrics (RMSE, MAE, R-squared) between the
simulated counts and ground-truth observations. The ``return_func_zoo``
dictionary maps string keys to these metric functions so that callers can
select a metric by name at runtime.
"""

import math
import numpy as np

from models.states import STATES
from sklearn.metrics import mean_squared_error, mean_absolute_error


def detected_active_counts(model, fit_column):
    """Extract the time-series of simulated counts for a given model column.

    Reads the model's output DataFrame and returns all rows after the first
    (index 0) as a NumPy array, matching the convention used by the
    optimisation routines.

    Args:
        model: A ``ModelM`` instance that has already been run.  Must expose
            a ``get_df()`` method returning a pandas DataFrame with at least
            one column named ``fit_column``.
        fit_column (str): Name of the DataFrame column to extract (e.g.
            ``'I_d'`` for detected infectious individuals).

    Returns:
        numpy.ndarray: 1-D array of simulated counts, one entry per
        simulated day (the day-0 initialisation row is excluded).
    """
    counts = model.get_df()
    counts = counts[fit_column][1:].to_numpy()

    return counts


def model_rmse(model, y_true, fit_column='I_d'):
    """Compute the root-mean-squared error between model output and observations.

    Args:
        model: A ``ModelM`` instance that has already been run.
        y_true (numpy.ndarray): Array of observed (ground-truth) values with
            the same length as the model output series.
        fit_column (str): Column in the model DataFrame to compare against
            ``y_true``.  Defaults to ``'I_d'``.

    Returns:
        float: RMSE between the model-predicted counts and ``y_true``.
    """
    infected_count = detected_active_counts(model, fit_column)
    return math.sqrt(mean_squared_error(y_true, infected_count))


def model_mae(model, y_true, fit_column='I_d'):
    """Compute the mean absolute error between model output and observations.

    Args:
        model: A ``ModelM`` instance that has already been run.
        y_true (numpy.ndarray): Array of observed (ground-truth) values with
            the same length as the model output series.
        fit_column (str): Column in the model DataFrame to compare against
            ``y_true``.  Defaults to ``'I_d'``.

    Returns:
        float: MAE between the model-predicted counts and ``y_true``.
    """
    infected_count = detected_active_counts(model, fit_column)
    return mean_absolute_error(y_true, infected_count)


def model_r_squared(model, y_true, fit_column='I_d'):
    """Compute a negated coefficient of determination (R²) for use as a loss.

    The value is negated so that minimising it corresponds to maximising the
    conventional R² fit quality.  A small epsilon (``np.finfo(np.float32).eps``)
    is added to the total sum of squares to avoid division by zero when
    ``y_true`` is constant.

    Args:
        model: A ``ModelM`` instance that has already been run.
        y_true (numpy.ndarray): Array of observed (ground-truth) values with
            the same length as the model output series.
        fit_column (str): Column in the model DataFrame to compare against
            ``y_true``.  Defaults to ``'I_d'``.

    Returns:
        float: Negated R² score, i.e. ``-(1 - RSS/TSS)``.  A perfect fit
        returns ``0.0``; worse fits return increasingly negative values.
    """
    infected_count = detected_active_counts(model, fit_column)
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2) + np.finfo(np.float32).eps
    rss = np.sum((y_true - infected_count) ** 2)
    return -(1 - rss / tss)


return_func_zoo = {
    'rmse': model_rmse,
    'mae': model_mae,
    'r2': model_r_squared
}
