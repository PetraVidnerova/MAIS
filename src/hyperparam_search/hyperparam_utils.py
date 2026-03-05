"""High-level utilities for running hyperparameter searches on epidemic models.

This module acts as the entry-point for hyperparameter optimisation workflows.
It loads model and search configurations from disk, wires together the model
runner with the chosen search method (grid search, CMA-ES, …), and exposes
``run_hyperparam_search`` as the primary public API.

Internal helpers handle model duplication, parameter pre-processing (e.g.
``theta`` expansion, ``beta_A`` derivation from ``a_reduction``), and
selection of a return/loss function from ``eval_model.return_func_zoo``.
"""

import json
from functools import partial

from typing import Dict

from utils.config_utils import ConfigFile
from hyperparam_search.eval_model import return_func_zoo
from hyperparam_search.search_methods import hyperparam_search_zoo
from model_m.model_m import load_model_from_config, load_graph


def run_hyperparam_search(model_config: str,
                          hyperparam_config: str,
                          model_random_seed: int = 42,
                          run_n_times: int = 1,
                          start_day: int = None,
                          n_days: int = None,
                          return_func: str = None,
                          return_func_kwargs: Dict = None,
                          **kwargs):
    """Run hyperparameter search on a model loaded from config.

    Hyperparameters specified in ``hyperparam_config`` overwrite those in
    ``model_config``.  The search method is also defined in
    ``hyperparam_config``.

    A single model run returns the model as a whole.  If only a subset of
    information is to be extracted, pass ``return_func`` – a string key that
    selects a specific metric function from ``eval_model.return_func_zoo``.

    Args:
        model_config (str): Path to the model configuration file (INI format).
        hyperparam_config (str): Path to the hyperparameter search
            configuration file (JSON format).
        model_random_seed (int): Initial random seed for every model.
            Defaults to ``42``.
        run_n_times (int): For a single model (with specific hyperparams),
            repeat the run multiple times.  The random seed for the i-th run
            is set to ``model_random_seed + i``.  Defaults to ``1``.
        start_day (int or None): Starting day for the experiment.  If
            ``None``, ``start_day`` from the model config is used.
        n_days (int or None): Number of days to run the experiment.  If
            ``None``, ``duration`` from the model config (or ``60`` if not
            set) is used.
        return_func (str or None): String key of the return/loss function
            from ``eval_model.return_func_zoo``.  ``None`` returns the model
            object itself.
        return_func_kwargs (Dict or None): Additional keyword arguments
            forwarded to the return function selected by ``return_func``.
        **kwargs: Additional keyword arguments forwarded to the hyperparameter
            search function (interpretation is method-specific).

    Returns:
        Result of hyperparameter optimisation, specific to the chosen search
        method (e.g. a list of dicts for grid search, or a single best-params
        dict for CMA-ES).
    """

    cf = ConfigFile()
    cf.load(model_config)

    if start_day is not None:
        cf.config['MODEL']['start_day'] = str(start_day)

    graph = load_graph(cf)
    base_model = load_model_from_config(cf, model_random_seed, preloaded_graph=graph)

    # wrapper for running one model same time with different seed
    model_load_func = partial(_run_models_from_config,
                              cf,
                              preloaded_graph=graph,
                              preloaded_model=base_model,
                              model_random_seed=model_random_seed,
                              run_n_times=run_n_times,
                              n_days=n_days,
                              return_func=return_func,
                              return_func_kwargs=return_func_kwargs)

    hyperparam_search_func = _init_hyperparam_search(hyperparam_config)
    return hyperparam_search_func(model_func=model_load_func, **kwargs)


def run_single_model(model, T, print_interval=10, verbose=False):
    """Run a single model for ``T`` days and return it.

    Args:
        model: A ``ModelM`` instance that is ready to be run (or will be set
            up automatically on the first call to ``model.run``).
        T (int): Number of simulation days.
        print_interval (int): How often (in days) to print a progress summary.
            Defaults to ``10``.
        verbose (bool): Whether to print detailed per-step output.
            Defaults to ``False``.

    Returns:
        ModelM: The same ``model`` object after simulation has completed.
    """
    model.run(T=T, verbose=verbose, print_interval=print_interval)
    return model


def _run_models_from_config(cf: ConfigFile,
                            preloaded_graph: None,
                            preloaded_model: None,
                            hyperparams: Dict = None,
                            model_random_seed: int = 42,
                            run_n_times: int = 1,
                            n_days: int = None,
                            return_func: str = None,
                            return_func_kwargs: Dict = None):
    """Run the model (possibly multiple times) with a specific set of hyperparameters.

    This is the inner callable that is partially applied by
    ``run_hyperparam_search`` and then passed to the search method.  It
    handles a handful of model-specific parameter transformations before
    constructing or duplicating the model:

    * ``theta`` is expanded into ``theta_E``, ``theta_Ia``, and ``theta_In``.
    * ``a_reduction`` is converted to ``beta_A = beta * a_reduction``.
    * ``beta`` / ``beta_A`` are mirrored into their ``*_in_family`` variants.
    * Keys starting with ``policy_`` are stripped of their prefix and forwarded
      to the policy constructor.

    Args:
        cf (ConfigFile): Loaded configuration object for the model.
        preloaded_graph: Pre-built graph object to avoid reloading from disk.
            Pass ``None`` to load from config.
        preloaded_model: A ``ModelM`` instance to duplicate instead of
            constructing a new model from scratch.  Pass ``None`` to build
            from config.
        hyperparams (Dict): Dictionary of hyperparameter names to values for
            this particular evaluation.  Defaults to ``None``.
        model_random_seed (int): Base random seed.  Defaults to ``42``.
        run_n_times (int): Number of independent repetitions (each with seed
            ``model_random_seed + i``).  Defaults to ``1``.
        n_days (int): Override for the simulation duration.  If ``None``,
            the value from config (or ``60``) is used.
        return_func (str): Key into ``return_func_zoo`` selecting the metric
            to compute.  ``None`` returns the model object itself.
        return_func_kwargs (Dict): Extra keyword arguments forwarded to the
            selected return function.

    Returns:
        dict: A dictionary with the following keys:

            * ``"result"`` – scalar metric (or list of metrics when
              ``run_n_times > 1``), or the model object when
              ``return_func`` is ``None``.
            * ``"hyperparams"`` – the (potentially transformed) hyperparameter
              dictionary used for this run.
            * ``"seed"`` – the base random seed that was used.
    """
    # copy model
    ndays = n_days if n_days is not None else cf.section_as_dict("TASK").get("duration", 60)
    print_interval = cf.section_as_dict("TASK").get("print_interval", 1)
    verbose = cf.section_as_dict("TASK").get("verbose", "Yes") == "Yes"

    # these are some special hacks related to infection model
    if "theta" in hyperparams:
        hyperparams.update({"theta_E": hyperparams["theta"],
                            "theta_Ia": hyperparams["theta"],
                            "theta_In": hyperparams["theta"]
                            })
        del hyperparams["theta"]

    if "a_reduction" in hyperparams:
        hyperparams["beta_A"] = hyperparams["beta"] * hyperparams["a_reduction"]
        del hyperparams["a_reduction"]

    if "beta" in hyperparams:
        hyperparams["beta_in_family"] = hyperparams["beta"]
    if "beta_A" in hyperparams:
        hyperparams["beta_A_in_family"] = hyperparams["beta_A"]

    policy_hyperparams = {k.replace('policy_', ''): v for k, v in hyperparams.items() if k.startswith('policy_')}
    hyperparams = {k: v for k, v in hyperparams.items() if not k.startswith('policy_')}

    if preloaded_model is None:
        model = load_model_from_config(cf, model_random_seed, preloaded_graph=preloaded_graph,
                                       hyperparams=hyperparams, policy_params=policy_hyperparams)
    else:
        model = preloaded_model.duplicate(model_random_seed, hyperparams, policy_params=policy_hyperparams)

    # for different seeds
    def _run_one_model(seed):
        model.reset(random_seed=seed)
        ret = run_single_model(model, T=ndays, print_interval=print_interval, verbose=verbose)
        return ret

    # specific return function or identity
    func = return_func_zoo[return_func] if return_func is not None else lambda m, **kwargs: m

    if run_n_times > 1:
        # add 1 to seed each run
        res = [func(_run_one_model(seed), **return_func_kwargs)
               for seed in range(model_random_seed, model_random_seed + run_n_times)]
    else:
        res = func(_run_one_model(model_random_seed), **return_func_kwargs)

    # optionally return additional run info
    return {
        "result": res,
        "hyperparams": hyperparams,
        "seed": model_random_seed
    }


def _init_hyperparam_search(hyperparam_file: str):
    """Load a hyperparameter search configuration and return the search callable.

    Reads a JSON file whose ``"method"`` key selects the search strategy from
    ``hyperparam_search_zoo``.  The full config dict is partially applied as
    the ``hyperparam_config`` argument of the chosen strategy function.

    Args:
        hyperparam_file (str): Path to the JSON configuration file describing
            the hyperparameter search (method, parameter ranges, CMA settings,
            etc.).

    Returns:
        functools.partial: A callable ``(model_func, **kwargs) -> result`` that
        executes the selected search strategy with the loaded configuration
        already bound.
    """
    with open(hyperparam_file, 'r') as json_file:
        config = json.load(json_file)

    return partial(hyperparam_search_zoo[config["method"]], hyperparam_config=config)
