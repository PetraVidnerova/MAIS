"""Factory for creating custom epidemic network-model classes at runtime.

This module provides :func:`create_custom_model`, which dynamically builds a
Python class that combines a user-supplied model definition (states,
transitions, parameters, propensity function) with a chosen simulation engine
(default: :class:`~models.engine_seirspluslike.SeirsPlusLikeEngine`).
"""

from models.engine_seirspluslike import SeirsPlusLikeEngine


def not_implemented_yet():
    """Placeholder that raises NotImplementedError.

    Used as a sentinel default for ``calc_propensities`` when none is supplied
    to :func:`create_custom_model`.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError()


def create_custom_model(clsname, states, state_str_dict, transitions,
                        final_states=[], invisible_states=[],
                        unstable_states=[],
                        init_arguments={},
                        model_parameters={},
                        calc_propensities=not_implemented_yet,
                        member_functions=None,
                        engine=SeirsPlusLikeEngine):
    """Dynamically create an epidemic model class from a model definition.

    Builds and returns a new Python class named *clsname* that inherits from
    *engine* and is populated with the supplied model metadata, parameters, and
    optional member functions.  The generated class receives an ``__init__``
    that accepts the contact graph ``G`` and keyword arguments corresponding to
    all declared parameters.

    Args:
        clsname (str): Name of the class to create.
        states (list): Ordered list of integer state codes.
        state_str_dict (dict): Mapping from state code (int) to label (str).
        transitions (list): List of ``(from_state, to_state)`` tuples that
            describe all allowed state transitions.
        final_states (list, optional): States from which no further transition
            occurs (absorbing states). Defaults to ``[]``.
        invisible_states (list, optional): States whose members are excluded
            from the active population count (e.g. deceased). Defaults to
            ``[]``.
        unstable_states (list, optional): States that can still change; used
            to decide when to terminate a simulation run. Defaults to ``[]``,
            which is treated as *all* states.
        init_arguments (dict, optional): Fixed constructor arguments as
            ``{name: (default, description)}``. Defaults to ``{}``.
        model_parameters (dict, optional): Per-node model parameters as
            ``{name: (default, description)}``.  Values may be scalars or
            arrays of length ``num_nodes``. Defaults to ``{}``.
        calc_propensities (callable, optional): Function with signature
            ``calc_propensities(model)`` that returns a list of per-node
            propensity arrays.  Used by SEIRS-Plus-Like derived engines.
            Defaults to :func:`not_implemented_yet`.
        member_functions (dict, optional): Additional methods to attach to
            the generated class, as ``{method_name: function}``. Defaults to
            ``None``.
        engine (type, optional): Base engine class.  Defaults to
            :class:`~models.engine_seirspluslike.SeirsPlusLikeEngine`.

    Returns:
        type: A new class inheriting from *engine* with all model metadata
        set as class variables and a suitable ``__init__`` attached.
    """

    # dictionary of future class variables
    attributes = {
        "states": states,
        "num_states": len(states),
        "state_str_dict": state_str_dict,
        "transitions": transitions,
        "num_transitions": len(transitions),
        "final_states": final_states,
        "invisible_states": invisible_states,
        "unstable_states": unstable_states or states,
        "fixed_model_parameters": init_arguments,
        "model_parameters": model_parameters,
        "common_arguments": {"random_seed": (None, "random seed value")}
    }

    if calc_propensities is None:
        calc_propensities = not_implemented_yet
    else:
        #model_cls.calc_propensities = calc_propensities
        attributes["calc_propensities"] = calc_propensities

    if member_functions is not None:
        assert type(member_functions) == dict
        attributes.update(member_functions)

    model_cls = type(clsname, (engine,), attributes)
    doc_text = """    A class to simulate the Stochastic Network Model

    Params:
            G       Network adjacency matrix (numpy array) or Networkx graph object \n"""

    for argname in ("fixed_model_parameters",
                    "model_parameters",
                    "common_arguments"):
        for param, definition in attributes[argname].items():
            param_text = f"            {param}       {definition[1]}\n"
            if argname == "model_parameters":
                param_text += f"            (float or np.array)\n"
            doc_text = doc_text + param_text

    model_cls.__doc__ = doc_text

    # __init__ method
    def init_function(self, G, **kwargs):
        """Initialise the model on contact graph *G*.

        Args:
            G: Contact network.  Accepted types: ``numpy.ndarray``,
                ``scipy.sparse.csr_matrix``, or ``networkx.Graph``.
            **kwargs: Keyword arguments that override any default declared in
                ``fixed_model_parameters``, ``model_parameters``, or
                ``common_arguments``.  State initial counts can be supplied as
                ``init_<STATE_LABEL>=<count>``.
        """
        # 1. set member variables acording to init arguments
        # definition is couple (default value, description)
        self.G = G
        self.graph = G
        self.A = None
        self.init_kwargs = kwargs

        # 2. model initialization
        self.inicialization()

        # 3. time and history setup
        self.setup_series_and_time_keeping()

        # 4. init states and their counts
        # print(self.init_state_counts)
        self.states_and_counts_init()

        # 5. set callback to None
        self.periodic_update_callback = None

    # add __init__
    model_cls.__init__ = init_function

    # # add member functions
    # function_list = [inicialization,
    #                  update_graph,
    #                  node_degrees,
    #                  setup_series_and_time_keeping,
    #                  states_and_counts_init,
    #                  set_periodic_update,
    #                  update_scenario_flags,
    #                  num_contacts,
    #                  current_state_count,
    #                  current_N,
    #                  run_iteration,
    #                  run,
    #                  finalize_data_series,
    #                  increase_data_series_length]

    # for function in function_list:
    #     setattr(model_cls, function.__name__, function)

    # def not_implemented_yet(self):
    #     raise NotImplementedError
    return model_cls
