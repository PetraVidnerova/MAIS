"""Policy and scenario loading utilities for the MAIS simulation.

This module provides functions for loading simulation scenario definitions
from CSV files and converting them to dictionaries that can drive per-day
policy changes (such as contact-layer weight adjustments) during a run.
"""

import pickle

import click
import pandas as pd


def load_scenario_dict(filename: str, sep=',', return_data='list'):
    """Load a scenario definition from a CSV file and return it as a dictionary.

    The CSV file is expected to have at minimum an ``id`` column, a ``name``
    column, and one or more numeric day-columns. Each day-column represents a
    particular simulation day, and the rows represent the layers (or other
    entities) whose weights/values are being specified for that day.

    Args:
        filename (str): Path to the CSV file to read.
        sep (str, optional): Column delimiter used in the CSV file.
            Defaults to ``','``.
        return_data (str, optional): Format for the per-day values:

            - ``'list'``: Each day maps to a plain Python list of values
              ordered by row.
            - ``'names'``: Each day maps to a ``{name: value}`` dictionary
              keyed by the ``name`` column.
            - ``'ids'``: Each day maps to a ``{name: value}`` dictionary
              keyed by the ``name`` column (same as ``'names'`` in current
              implementation).

            Defaults to ``'list'``.

    Returns:
        dict: Mapping of ``{day (int): output}`` where ``output`` is
        formatted according to ``return_data``.

    Raises:
        ValueError: If ``return_data`` is not one of ``'list'``,
            ``'names'``, or ``'ids'``.
    """
    df = pd.read_csv(filename, sep=sep)

    def get_output(series: pd.Series):
        l = series.to_list()

        if return_data == 'list':
            return l
        elif return_data == 'names':
            return {k: v for k, v in zip(df["name"], l)}
        elif return_data == 'ids':
            return {k: v for k, v in zip(df["name"], l)}
        else:
            raise ValueError(f"Unsupported format: {return_data}, valid options are 'list', 'names', 'ids'.")

    data_only = df.drop(columns=['id', 'name'])
    return {int(k): get_output(data_only[k]) for k in data_only.columns}


@click.command()
@click.argument('filename')
@click.option('--out_path', default=None)
@click.option('--sep', default=',')
@click.option('--return_data', default='list')
def run(filename, out_path, sep, return_data):
    """CLI entry point: load a scenario CSV and optionally serialise it.

    Reads the scenario defined in ``filename``, prints the resulting
    dictionary to stdout, and if ``--out_path`` is provided, serialises the
    dictionary to that path using :mod:`pickle`.

    Args:
        filename (str): Path to the scenario CSV file (positional argument).
        out_path (str or None): Optional file path for the pickled output.
            Defaults to ``None`` (no file written).
        sep (str): Column delimiter for the CSV. Defaults to ``','``.
        return_data (str): Output format passed to
            :func:`load_scenario_dict`. Defaults to ``'list'``.
    """
    out_dict = load_scenario_dict(filename, sep=sep, return_data=return_data)

    if out_path is not None:
        with open(out_path, 'wb') as of:
            pickle.dump(out_dict, of)

    print(out_dict)


if __name__ == "__main__":
    run()
