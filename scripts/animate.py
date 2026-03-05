"""Interactive animation of per-node epidemic states using graph-tool.

This script visualises a pre-computed node-state time-series (produced by
enabling ``save_node_states`` in a simulation run) as an animated graph
rendered with ``graph-tool`` and ``GTK``.

Each node is rendered as a gender-specific icon (lady / man); nodes that
are in an infectious state are highlighted with a red halo.  Nodes that die
are replaced with a zombie icon.  The animation advances one day every two
seconds.

Requires ``graph-tool``, ``cairo``, and the GTK bindings to be installed.

Typical usage::

    python animate.py config.ini --nodes_file ../data/output/model/node_states.csv
"""

import sys
import os
import click
import time
import pandas as pd
import graph_tool.all as gt

import cairo
from gi.repository import Gtk, GLib

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))


from utils.config_utils import ConfigFile



@click.command()
@click.argument('filename')
@click.option('--nodes_file', default="../data/output/model/node_states.csv")
def main(filename, nodes_file):
    """Animate a node-state simulation result as an interactive graph.

    Reads node and edge CSV files referenced in ``filename``'s ``[GRAPH]``
    section, builds a ``graph-tool`` graph with node icons and edge widths
    derived from the original contact-network data, and then plays back the
    per-node state time-series loaded from ``nodes_file``.

    Infectious nodes are shown with a red halo; deceased nodes switch to a
    zombie icon.  The GTK event loop is used to advance the animation one
    day at a time (with a two-second pause between days).

    Args:
        filename (str): Path to the experiment INI configuration file.  The
            ``[GRAPH]`` section must specify ``nodes`` and ``edges`` CSV
            paths.
        nodes_file (str): Path to the per-node state CSV produced by the
            simulation (rows = nodes, columns = days).  Defaults to
            ``"../data/output/model/node_states.csv"``.
    """
    cf = ConfigFile()
    cf.load(filename)

    nodes_filename = cf.section_as_dict("GRAPH")["nodes"]
    edges_filename = cf.section_as_dict("GRAPH")["edges"]

    nodes = pd.read_csv(nodes_filename)[["id", "label", "sex"]]
    edges = pd.read_csv(edges_filename)[["vertex1", "vertex2", "probability", "intensity"]]

    g = gt.Graph(directed=False)
    node_label = g.new_vertex_property("string")
    node_sex = g.new_vertex_property("object")

    edge_proba = g.new_edge_property("float")
    edge_intensity = g.new_edge_property("float")

    lady = cairo.ImageSurface.create_from_png("fig/lady.png")
    man = cairo.ImageSurface.create_from_png("fig/man.png")
    dead = cairo.ImageSurface.create_from_png("fig/zombie.png")

    node_state_I = g.new_vertex_property("bool")
    node_state_R = g.new_vertex_property("bool")

    df = pd.read_csv(nodes_file, index_col=0).transpose()
    max_day = df.shape[1] - 1     
    
    def update_states(day):
        """Update graph-tool vertex properties for a given simulation day.

        Sets the infectious-halo flag for each node based on the state column
        for ``day``.  Nodes whose state equals ``2`` (deceased) have their
        surface icon replaced with the ``dead`` image.

        Args:
            day (int): The column index in ``df`` representing the simulation
                day to visualise.
        """
        for i, row in nodes.iterrows():
            node_state_I[i] = df.loc[str(row["id"]), day] == 1
            if df.loc[str(row["id"]), day] == 2:
                node_sex[i] = dead



    for i, row in nodes.iterrows():
        v = g.add_vertex()
        node_label[v] = row["label"]
        node_sex[v] = lady if row["sex"] == "F" else man
        update_states(0)

    for i, row in edges.iterrows():
        v1, v2 = row["vertex1"], row["vertex2"]
        proba = row["probability"]
        intens = row["intensity"]

        e = g.add_edge(v1, v2, add_missing=True)
        edge_proba[e] = proba
        edge_intensity[e] = intens
        edge_width = proba*intens*2
        
        node_label[e.source()] = nodes.loc[nodes["id"] == v1, "label"].values[0]
        node_label[e.target()] = nodes.loc[nodes["id"] == v2, "label"].values[0]
        


    win = gt.GraphWindow(g, 
        pos = gt.sfdp_layout(g),
        geometry=(1200, 800),
        vertex_size=30,
        vertex_anchor=0,
        edge_color="gray",
        edge_pen_width =edge_width,
        edge_sloppy=True,
        vertex_surface=node_sex,
        vertex_color=[1.,1.,1.,0.],
        vertex_fill_color=[1.,1.,1.,0.],
        vertex_halo=node_state_I,
        vertex_halo_size=1.2,
        vertex_halo_color=[0.8, 0, 0, 0.6]
    )

    class Day():
        """Mutable container for the current animation day index.

        Used as a closure-friendly counter so that the ``update_state``
        callback can advance the day without relying on a mutable outer-scope
        variable.

        Args:
            day (int): Initial day index (0-based).
        """

        def __init__(self, day):
            self.day = day

        def increase(self):
            """Increment the day counter by one."""
            self.day += 1
            
    day = Day(0)

    def update_state(day):
        """GTK idle callback that advances the animation by one day.

        Calls ``update_states`` for the current day, redraws the graph
        window, sleeps for two seconds, and then increments the day counter.

        Args:
            day (Day): Mutable day counter object.

        Returns:
            bool: ``True`` to keep the idle callback registered (continue
            animation); ``False`` when the last simulation day has been
            reached, which removes the callback and stops the animation.
        """
        print("Day", day.day)
        update_states(day.day)

        win.graph.regenerate_surface()
        win.graph.queue_draw()
        day.increase()
        time.sleep(2)
        if day.day > max_day:
            print("End of simulation")
            return False
        return True


    cid = GLib.idle_add(lambda: update_state(day))

    # We will give the user the ability to stop the program by closing the window.
    win.connect("delete_event", Gtk.main_quit)

    # Actually show the window, and start the main loop.
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
