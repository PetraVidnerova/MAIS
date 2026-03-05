"""Model-M wrapper and graph loading utilities for the MAIS epidemic simulator.

This module provides:

* ``ModelM`` – a high-level wrapper around the low-level epidemic models
  (selected via ``model_zoo``).  It manages graph copying, adjacency-matrix
  initialisation, policy attachment, and multi-run resets.
* ``load_model_from_config`` – convenience factory that builds a ``ModelM``
  instance from a ``ConfigFile`` object.
* ``load_graph`` / ``save_graph`` – helpers for constructing or persisting
  graph objects.
* ``_load_policy_function`` – internal helper that dynamically imports and
  configures the policy class named in the config file.
"""

import os
import pickle

import numpy as np
from utils.config_utils import ConfigFile

from graphs.graph_gen import GraphGenerator, CSVGraphGenerator, RandomSingleGraphGenerator
from graphs.light import LightGraph
from graphs.simple import SimpleGraph
from models.model_zoo import model_zoo
from models.states import STATES

import logging


def load_model_from_config(cf, model_random_seed, preloaded_graph=None, hyperparams=None, policy_params=None):
    """Construct a ``ModelM`` instance from a loaded ``ConfigFile``.

    Reads model parameters, graph specification, policy definition, and
    scenario from the supplied config, then assembles and returns a
    ``ModelM`` object (not yet run).

    Args:
        cf (ConfigFile): Loaded configuration object containing at least
            ``[MODEL]``, ``[GRAPH]``, and ``[TASK]`` sections.
        model_random_seed (int): Random seed to pass to the underlying model.
        preloaded_graph: A pre-built graph object.  When provided, graph
            loading from config is skipped.  Defaults to ``None``.
        hyperparams (dict or None): Additional model parameters that override
            values from ``[MODEL]``.  Defaults to ``None``.
        policy_params (dict or None): Additional policy constructor parameters
            that override values from ``[POLICY_SETUP]``.  Defaults to
            ``None``.

    Returns:
        ModelM: A freshly constructed (but not yet set up or run) model
        instance.
    """
    # load model hyperparameters
    model_params = cf.section_as_dict("MODEL")
    if hyperparams is not None:
        model_params = {**model_params, **hyperparams}

    # load graph as described in config file
    if preloaded_graph is not None:
        graph = preloaded_graph
    else:
        graph = load_graph(cf)

    #  policy
    policy = _load_policy_function(cf, policy_params=policy_params)

    # sceanario
    scenario = cf.section_as_dict("SCENARIO")
    scenario = scenario["closed"] if scenario else None

    # model type
    model_type = cf.section_as_dict("TASK").get(
        "model", "SimulationDrivenModel")

    model = ModelM(graph,
                   policy,
                   model_params,
                   scenario,
                   random_seed=model_random_seed,
                   model_type=model_type
                   )
    return model


class ModelM():
    """High-level wrapper around a MAIS epidemic simulation model.

    ``ModelM`` acts as a facade that combines a contact graph, an optional
    non-pharmaceutical intervention policy, epidemic model parameters, and a
    layer-closure scenario into a single object with a simple
    ``run`` / ``reset`` / ``duplicate`` interface.

    The underlying low-level model is instantiated lazily (on the first call to
    ``run`` or ``reset``) via ``setup()``.

    Args:
        graph: A graph object (``LightGraph``, ``SimpleGraph``, or
            ``GraphGenerator`` subclass) describing the contact network.  A
            working copy is created for each simulation run.
        policy: A two-element tuple ``(PolicyClass, setup_dict)`` where
            ``PolicyClass`` is the policy constructor and ``setup_dict``
            contains keyword arguments for it.  Either element may be
            ``None`` to run without a policy.
        model_params (dict or None): Epidemic model parameters forwarded to
            the low-level model constructor.  Defaults to ``None``.
        scenario (list or None): List of layer names to close before building
            the adjacency matrix.  Only meaningful for ``GraphGenerator``
            graphs.  Defaults to ``None``.
        model_type (str): Key into ``model_zoo`` selecting the low-level model
            class.  Defaults to ``'SimulationDrivenModel'``.
        random_seed (int): Initial random seed passed to the underlying model.
            Defaults to ``42``.
    """

    def __init__(self,
                 graph,
                 policy,
                 model_params: dict = None,
                 scenario: list = None,
                 model_type: str = "SimulationDrivenModel",
                 random_seed: int = 42):

        # self.random_seed = 42

        # original state
        self.start_graph = graph

        # scenario (list of closed layers)
        self.scenario = scenario

        self.model_type = model_type
        self.model_params = model_params
        self.random_seed = random_seed
        self.policy = policy
        self.policy_object = None

        self.model = None
        self.ready = False

        self.graph = None
        self.A = None

    def setup(self):
        """Initialise the model for a fresh simulation run.

        Creates a working copy of the graph, builds the adjacency matrix,
        instantiates the low-level epidemic model and (if configured) the
        policy object.  Sets ``self.ready = True`` when complete.

        Any previously allocated graph, adjacency matrix, model, or policy
        objects are deleted before re-initialisation to free memory.
        """
        logging.info("model setup")
        # working copy of graph and matrix
        if self.graph is not None:
            del self.graph
        if self.A is not None:
            del self.A
        if self.model is not None:
            del self.model

        if self.policy_object is not None:
            self.policy_object.graph = None
            self.policy_object.model = None
            del self.policy_object

        self.graph = self.start_graph.copy()
        self.A = self.init_matrix()

        # model
        Model = model_zoo[self.model_type]
        self.model = Model(self.A,
                           **self.model_params,
                           random_seed=self.random_seed)
        if self.policy[0] is not None:
            self.policy_object = self.policy[0](
                self.graph, self.model, **self.policy[1])
        else:
            self.policy_object = None
        self.model.set_periodic_update(self.policy_object)
        self.ready = True

    def duplicate(self, random_seed=None, hyperparams=None, policy_params=None):
        """Create a shallow copy of this model with optionally overridden settings.

        Only models that have not yet been set up (``self.ready == False``) can
        be duplicated.  The duplicate shares the original ``start_graph``
        object but builds its own working graph copy and adjacency matrix.

        Args:
            random_seed (int or None): Random seed for the duplicate.  If
                ``None``, the original seed is reused.
            hyperparams (dict or None): Overrides for model parameters.  Merged
                on top of the original ``model_params``.
            policy_params (dict or None): Overrides for policy constructor
                parameters.  Merged on top of the original policy setup dict.

        Returns:
            ModelM: A new ``ModelM`` instance with ``ready = False`` that can
            be run independently.

        Raises:
            NotImplementedError: If called on a model that has already been set
                up (``self.ready == True``).
        """
        #        pritn("DBD duplicate")

        if self.ready:
            raise NotImplementedError("We duplicate only newbie models")

        if policy_params:
            policy_func, policy_setup = self.policy
            policy_setup = {**policy_setup, **policy_params}
            policy = policy_func, policy_setup
        else:
            policy = self.policy

        twin = ModelM(
            self.start_graph,
            policy,
            self.model_params if hyperparams is None else dict(
                self.model_params, **hyperparams),
            self.scenario,
            random_seed=self.random_seed if random_seed is None else random_seed,
            model_type=self.model_type
        )
        twin.graph = twin.start_graph.copy()
        twin.A = twin.init_matrix()
        twin.ready = False
        return twin

    def set_model_params(self, model_params: dict):
        """Update the underlying model's parameters after construction.

        Args:
            model_params (dict): Dictionary of parameter names to new values
                forwarded to the low-level model's ``setup_model_params``
                method.
        """
        self.model.setup_model_params(model_params)

    def run(self, *args, **kwargs):
        """Run the simulation, setting up the model first if necessary.

        If the model has not been set up yet (``self.ready == False``),
        ``setup()`` is called automatically before delegating to the
        underlying model's ``run`` method.

        Args:
            *args: Positional arguments forwarded to the underlying model's
                ``run`` method (e.g. ``T`` for the number of days).
            **kwargs: Keyword arguments forwarded to the underlying model's
                ``run`` method (e.g. ``verbose``, ``print_interval``).
        """
        if not self.ready:
            self.setup()
        self.model.run(*args, **kwargs)

    def reset(self, random_seed=None):
        """Reset the model to its initial state for a new simulation run.

        If the model has never been set up, ``setup()`` is called first.
        Otherwise, the working graph and adjacency matrix are recreated from
        the original ``start_graph``, the policy object is rebuilt, and the
        low-level model's initialisation routines are re-invoked.

        The random seed can be overridden at reset time to produce an
        independent trajectory.

        Args:
            random_seed (int or None): New random seed.  If ``None``, the
                seed set during construction is retained.
        """
        #        print("DBD reset")
        if not self.ready:
            self.setup()
        else:
            del self.graph
            self.graph = self.start_graph.copy()
            del self.A
            self.A = self.init_matrix()
            self.model.update_graph(self.graph)

            if self.policy_object is not None:
                del self.policy_object
            if self.policy[0] is not None:
                self.policy_object = self.policy[0](
                    self.graph, self.model, **self.policy[1])
            else:
                self.policy_object = None
            self.model.set_periodic_update(self.policy_object)

        self.model.inicialization()

        # random_seed has to be setup AFTER inicialization and BEFORE states_and_counts_init !
        if random_seed:
            self.model.set_seed(random_seed)
            logging.debug(f"seed changed to {self.model.random_seed}")

        self.model.setup_series_and_time_keeping()
        self.model.states_and_counts_init(ext_nodes=self.model.num_ext_nodes,
                                          ext_code=STATES.EXT)

    def get_results(self,
                    states):
        """Return the time-series count(s) for one or more epidemic states.

        Args:
            states (str or list[str]): A single state name or a list of state
                names whose count time-series should be retrieved.

        Returns:
            numpy.ndarray or list[numpy.ndarray]: A single count array when
            ``states`` is a string, or a list of count arrays when ``states``
            is a list.
        """
        if type(states) == list:
            return [self.model.get_state_count(s) for s in states]
        else:
            return self.model.get_state_count(states)

    def get_df(self):
        """Return simulation results as a merged pandas DataFrame.

        The model DataFrame (one row per time step, columns for each epidemic
        state) is merged with the policy DataFrame (if a policy is active) on
        the ``"T"`` column.  Policy-specific columns that share a name with
        model columns are suffixed with ``"_policy"``.

        Returns:
            pandas.DataFrame: Merged DataFrame with at minimum a ``"T"``
            column and per-state count columns.
        """
        model_df = self.model.to_df()
        if self.policy_object is not None:
            policy_df = self.policy_object.to_df()
        else:
            policy_df = None

        df = model_df.merge(policy_df, on="T",
                            how="outer", suffixes=("", "_policy")) if policy_df is not None else model_df
        return df

    def save_history(self, file_or_filename):
        """Save the full simulation history to a CSV file.

        Merges the model DataFrame with the policy DataFrame (same logic as
        ``get_df``) and writes the result as CSV.  The DataFrame is also
        printed to stdout.

        Args:
            file_or_filename (str or file-like): Destination path or an
                already-opened writable file object accepted by
                ``pandas.DataFrame.to_csv``.
        """
        model_df = self.model.to_df()
        if self.policy_object is not None:
            policy_df = self.policy_object.to_df()
        else:
            policy_df = None

        df = model_df.merge(policy_df, on="T",
                            how="outer", suffixes=("", "_policy")) if policy_df is not None else model_df

#        cols = df.columns.tolist()
#        print(cols)
#        cols.insert(0, cols.pop(cols.index('T')))
#        df = df.reindex(columns= cols)

        df.to_csv(file_or_filename)
        print(df)

    def save_node_states(self, filename):
        """Save per-node state time-series to a CSV file.

        Delegates directly to the underlying model's ``save_node_states``
        method.

        Args:
            filename (str): Destination file path.
        """
        self.model.save_node_states(filename)

    def init_matrix(self):
        """Build and return the adjacency matrix (or graph object) for the model.

        The return type depends on the concrete graph class:

        * ``LightGraph`` / ``SimpleGraph`` – the graph object itself is
          returned directly (the model accepts it natively).
        * ``RandomSingleGraphGenerator`` – returns the underlying NetworkX
          graph ``G`` (scenarios are not supported in this case).
        * ``GraphGenerator`` – closes any layers listed in ``self.scenario``
          and returns the final adjacency matrix.

        Returns:
            The adjacency representation accepted by the underlying epidemic
            model (either the graph object or a sparse/dense matrix).

        Raises:
            NotImplementedError: If ``RandomSingleGraphGenerator`` is used
                together with a scenario.
            TypeError: If the graph type is not recognised.
        """
        if isinstance(self.graph, LightGraph) or isinstance(self.graph, SimpleGraph):
            #            raise NotImplementedError(
            #                "LighGraph not  supported at the moment, waits for fixes.")
            return self.graph

        if isinstance(self.graph, RandomSingleGraphGenerator):
            if self.scenario:
                raise NotImplementedError(
                    "RandomGraphGenerator does not support layers.")
            return grahp.G

        # this is what we currently used
        if isinstance(self.graph, GraphGenerator):
            if self.scenario:
                self.graph.close_layers(self.scenario)
            return self.graph.final_adjacency_matrix()

        raise TypeError("Unknown type of graph")


def load_graph(cf: ConfigFile):
    """Load or construct the contact-network graph described in ``cf``.

    The graph type is determined by the ``type`` key in the ``[GRAPH]`` config
    section.  Supported types:

    * ``"light"`` – ``LightGraph`` loaded from CSV files.
    * ``"simple"`` – ``SimpleGraph`` loaded from an edge CSV.
    * ``"pickle"`` – a previously serialised graph is loaded with
      ``pickle.load``.

    If a ``file`` key is present in ``[GRAPH]`` and the file already exists on
    disk the type is automatically overridden to ``"pickle"``.  After loading a
    non-pickle graph, if a ``file`` path was specified the graph is serialised
    with ``save_graph`` for future reuse.

    Args:
        cf (ConfigFile): Loaded configuration object with a ``[GRAPH]``
            section containing at minimum a ``type`` key.

    Returns:
        LightGraph or SimpleGraph or GraphGenerator: The loaded graph object.

    Raises:
        NotImplementedError: If ``type`` is ``"csv"`` or ``"random"``
            (deprecated / unsupported).
        ValueError: If ``type`` is not one of the supported values.
        AssertionError: If a pickle file contains an unexpected object type.
    """
    logging.info("Load graph.")

    num_nodes = cf.section_as_dict("TASK").get("num_nodes", None)

    cf_graph = cf.section_as_dict("GRAPH")

    graph_type = cf_graph["type"]
    filename = cf_graph.get("file", None)
    nodes = cf_graph.get("nodes", "nodes.csv")
    edges = cf_graph.get("edges", "edges.csv")
    layers = cf_graph.get("layers", None)
    externals = cf_graph.get("externals", None)
    quarantine = cf_graph.get("quarantine", None)
    layer_groups = cf_graph.get("layer_groups", None)

    if filename is not None and os.path.exists(filename):
        graph_type = 'pickle'

    if graph_type == "csv":
        raise NotImplementedError(
            "Sorry. Graph_type 'csv' is not supported anymore. Use light graph.")
        # g = CSVGraphGenerator(path_to_nodes=nodes,
        #                       path_to_external=externals,
        #                       path_to_edges=edges,
        #                       path_to_layers=layers,
        #                       path_to_quarantine=quarantine)

#    if graph_name == "csv_light":
#        return LightGraph(path_to_nodes=nodes, path_to_edges=edges, path_to_layers=layers)

    elif graph_type == "light":
        g = LightGraph()
        g.read_csv(path_to_nodes=nodes,
                   path_to_external=externals,
                   path_to_edges=edges,
                   path_to_layers=layers,
                   path_to_quarantine=quarantine,
                   path_to_layer_groups=layer_groups)
        
    elif graph_type == "simple":
        g = SimpleGraph()
        g.read_csv(path_to_edges=edges)

    elif graph_type == "random":
        raise NotImplementedError(
            "Sorry. Graph_type 'random' is not supported now. Use light graph and stay tuned.")
        # g = RandomGraphGenerator()

    elif graph_type == "pickle":
        with open(filename, "rb") as f:
            g = pickle.load(f)
            if isinstance(g, GraphGenerator):
                if g.A_valid:
                    print("Wow, matrix A is ready.")
            else:
                assert isinstance(g, LightGraph) or isinstance(g, SimpleGraph), f"Something weird ({type(g)}) was loaded."
    else:
        raise ValueError(f"Graph {graph_type} not available.")

    if graph_type != "pickle" and filename is not None:
        save_graph(filename, g)
    return g


def save_graph(filename, graph):
    """Serialise a graph object to disk using pickle protocol 4.

    Args:
        filename (str): Destination file path.
        graph: Any graph object that is pickle-serialisable (e.g.
            ``LightGraph``, ``SimpleGraph``, ``GraphGenerator``).
    """
    with open(filename, 'wb') as f:
        pickle.dump(graph, f, protocol=4)


def _load_policy_function(cf: ConfigFile, policy_params=None):
    """Dynamically import and configure the policy class specified in the config.

    The function reads the ``[POLICY]`` section of ``cf`` for the class name
    (``name``) and the module file (``filename``), then imports
    ``policies.<filename>.<name>`` at runtime.  Policy constructor arguments
    are read from ``[POLICY_SETUP]`` and merged with any overrides supplied
    in ``policy_params``.

    If no ``name`` is specified in ``[POLICY]``, the function returns
    ``(None, None)``, signalling that no policy should be attached.

    Args:
        cf (ConfigFile): Loaded configuration object.
        policy_params (dict or None): Additional keyword arguments that
            override values from ``[POLICY_SETUP]``.  Defaults to ``None``.

    Returns:
        tuple: A two-element tuple ``(PolicyClass, setup_dict)`` where
        ``PolicyClass`` is the importted class (or ``None``) and
        ``setup_dict`` is the merged constructor keyword arguments (or
        ``None``).

    Raises:
        ValueError: If the ``[POLICY]`` section is present but the ``filename``
            key is missing.
    """
    policy_cfg = cf.section_as_dict("POLICY")

    policy_name = policy_cfg.get("name", None)
    if policy_name is None:
        return None, None

    if "filename" in policy_cfg:
        PolicyClass = getattr(
            __import__(
                "policies."+policy_cfg["filename"],
                globals(), locals(),
                [policy_name],
                0
            ),
            policy_name
        )
        setup = cf.section_as_dict("POLICY_SETUP")
        if policy_params is not None:
            setup = {**setup, **policy_params}

        return PolicyClass, setup
    else:
        print("Warning: NO POLICY IN CFG")
        print(policy_cfg)
        raise ValueError("Unknown policy.")
