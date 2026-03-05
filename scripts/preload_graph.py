"""Command-line script for pre-loading and pickling a contact-network graph.

This script reads a model configuration file, constructs the graph described
in the ``[GRAPH]`` section, and saves it as a pickle file at the path given
by the ``file`` key in ``[GRAPH]``.  On subsequent runs, ``load_graph``
automatically detects the pickle file and skips the (potentially expensive)
CSV parsing step.

Typical usage::

    python preload_graph.py config.ini
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import click
from utils.config_utils import ConfigFile

from model_m.model_m import load_graph, save_graph


@click.command()
@click.argument('filename', default="example.ini")
def main(filename):
    """ Load the graph and pickle. """

    cf = ConfigFile()
    cf.load(filename)

    filename = cf.section_as_dict("GRAPH").get("file", None)

    if filename is not None:
        graph = load_graph(cf)
    else:
        print("Please, specify path and name for the graph to save in you INI file (e.g. 'file=./graph.pickle').")
        print("Graph not loaded.")


if __name__ == "__main__":
    main()
