"""Hyperparameter search algorithms for epidemic model calibration.

This module implements two search strategies:

* **Grid search** (``perform_gridsearch``) – exhaustive evaluation of a
  Cartesian product of discrete parameter values using a multiprocessing pool.
* **CMA-ES** (``cma_es``) – a gradient-free evolutionary strategy (Covariance
  Matrix Adaptation Evolution Strategy) for continuous parameter spaces.

Both strategies accept a ``model_func`` callable with signature
``model_func(hyperparams: dict) -> dict`` and a ``hyperparam_config`` dict
loaded from a JSON configuration file.

Internal helpers manage sigmoid-based parameter rescaling so that CMA-ES
operates on an unconstrained real-valued search space.

The ``hyperparam_search_zoo`` dictionary maps string keys (``'GridSearch'``,
``'CMA-ES'``) to the corresponding search function.
"""

import warnings

import cma
import numpy as np
import scipy.special
from functools import partial
from multiprocessing import Pool

from cma.optimization_tools import EvalParallel2
from sklearn.model_selection import ParameterGrid
import gc


def _run_model_with_hyperparams(model_func, hyperparams, output_file=None):
    """Evaluate ``model_func`` with a specific hyperparameter dict and log the result.

    Args:
        model_func (callable): Function with signature
            ``model_func(hyperparams: dict) -> dict``.  The returned dict must
            have a ``"result"`` key whose value is a scalar or a sequence of
            scalars.
        hyperparams (dict): Hyperparameter values for this evaluation.
        output_file (str or None): Path to a CSV file for logging.  Each call
            appends one row.  Pass ``None`` to disable logging.

    Returns:
        dict: Raw result dict returned by ``model_func``, including at least
        ``"result"`` and ``"hyperparams"`` keys.
    """
    print(f"Running with hyperparams: {hyperparams}", flush=True)

    res = model_func(hyperparams=hyperparams)

    fitness = np.mean(res["result"])
    _log_inidividual(output_file, hyperparams.values(), fitness, 0)
    print(f"Finished run with hyperparams: {hyperparams}")
    return res


def perform_gridsearch(model_func, hyperparam_config, n_jobs=1, output_file=None):
    """Exhaustively evaluate all parameter combinations defined in ``hyperparam_config``.

    Builds a Cartesian product of the discrete parameter values listed under
    the ``"MODEL"`` key of ``hyperparam_config`` and evaluates each combination
    in parallel using a ``multiprocessing.Pool``.

    Args:
        model_func (callable): Function with signature
            ``model_func(hyperparams: dict) -> dict`` that runs the model and
            returns a result dictionary.
        hyperparam_config (dict): Search configuration.  Must contain a
            ``"MODEL"`` key mapping parameter names to lists of candidate
            values (same format as ``sklearn.model_selection.ParameterGrid``).
        n_jobs (int): Number of parallel worker processes.  Defaults to ``1``
            (sequential execution).
        output_file (str or None): Path to a CSV log file.  Pass ``None`` to
            disable logging.

    Returns:
        list[dict]: List of result dicts (one per parameter combination),
        each containing at least ``"result"`` and ``"hyperparams"`` keys.
    """
    grid = hyperparam_config["MODEL"]
    param_grid = ParameterGrid(grid)

    header = grid.keys()
    _init_output_file(output_file, header)

    run_model = partial(_run_model_with_hyperparams, model_func, output_file=output_file)
    with Pool(processes=n_jobs) as pool:
        res = pool.map(run_model, param_grid)
    return res


def evaluate_with_params(param_array: np.ndarray, model_func, param_keys, param_ranges=None):
    """Evaluate ``model_func`` using a raw (sigmoid-encoded) parameter array.

    Decodes ``param_array`` from the internal CMA-ES representation (applying
    the inverse sigmoid and optional range rescaling via ``_compile_individual``)
    and then calls ``model_func`` with the resulting hyperparameter dictionary.

    Args:
        param_array (numpy.ndarray): 1-D array of parameter values in the
            internal (unconstrained) search space used by CMA-ES.
        model_func (callable): Function with signature
            ``model_func(hyperparams: dict) -> dict``.
        param_keys (list[str]): Ordered list of parameter names corresponding
            to the elements of ``param_array``.
        param_ranges (dict or None): Optional mapping of parameter name to
            ``(lower, upper)`` bounds used for rescaling.  Pass ``None`` for
            the sigmoid-only (``[0, 1]``) encoding.

    Returns:
        float: Mean of the model result values (``np.mean(res["result"])``).

    Raises:
        AssertionError: If the length of ``param_array`` does not match the
            length of ``param_keys``.
    """
    assert len(param_array) == len(param_keys)

    hyperparam_dict = _compile_individual(param_array, param_keys=param_keys, param_ranges=param_ranges)
    model_res = model_func(hyperparams=hyperparam_dict)["result"]

    return np.mean(model_res)


def _keys_with_evolved_vals(evolved_vals, keys):
    """Zip parameter keys and values into a dictionary.

    Args:
        evolved_vals (iterable): Sequence of parameter values.
        keys (iterable): Sequence of parameter names, in the same order as
            ``evolved_vals``.

    Returns:
        dict: Mapping from parameter name to value.
    """
    return {k: v for k, v in zip(keys, evolved_vals)}


def _init_output_file(output_file, header):
    """Create or overwrite the CSV log file and write its header row.

    If ``output_file`` is ``None`` the function is a no-op.

    Args:
        output_file (str or None): Path to the CSV log file to create.
        header (iterable[str]): Ordered sequence of parameter names that form
            the middle columns of the header.  The row format is
            ``gen,<param1>,<param2>,...,fitness``.
    """
    if output_file is not None:
        with open(output_file, 'w+') as of:
            key_string = ','.join(header)
            of.write(f"gen,{key_string},fitness\n")


def _log_inidividual(output_file, x, fitness, gen):
    """Append one individual's evaluation record to the CSV log file.

    If ``output_file`` is ``None`` the function is a no-op.

    Args:
        output_file (str or None): Path to an already-initialised CSV log file.
        x (iterable): Sequence of decoded (human-readable) parameter values
            for this individual.
        fitness (float): Fitness (loss) value for this individual.
        gen (int): Generation index (0-based) at which this individual was
            evaluated.
    """
    if output_file is not None:
        with open(output_file, 'a') as of:
            of.write(f'{gen},{",".join(str(val) for val in x)},{fitness}\n')  # joined hyperparam values and fitness


def _inverse_sigmoid(x):
    """Apply the logit (inverse-sigmoid) function element-wise to ``x``.

    Values are clamped to ``[eps, 1 - eps]`` (where ``eps`` is
    ``np.finfo(np.float32).eps``) before the transform to avoid numerical
    instability at the boundaries.

    Args:
        x (numpy.ndarray): Array of values in the interval ``(0, 1)``.

    Returns:
        numpy.ndarray: Logit-transformed values (unconstrained real numbers).
    """
    x = np.clip(x, 0+np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
    return np.log(x/(1-x))


def _init_values(value_dict, ranges=None):
    """Convert initial parameter values to the internal CMA-ES search space.

    Parameters that have defined bounds in ``ranges`` are first linearly scaled
    to ``[0, 1]`` and then mapped through the logit transform.  Parameters
    without explicit bounds are assumed to already be in ``(0, 1)`` and are
    only logit-transformed.  Any parameter that appears in ``ranges`` but not
    in ``value_dict`` is initialised to ``0.5`` (the midpoint of the bounded
    interval before logit-transform).

    Args:
        value_dict (dict): Mapping of parameter name to initial value.
        ranges (dict or None): Optional mapping of parameter name to
            ``(lower, upper)`` bounds.  Parameters not present in ``ranges``
            must already lie in ``(0, 1)``.  Pass ``None`` to treat all
            parameters as unbounded in ``(0, 1)``.

    Returns:
        tuple: A two-element tuple ``(keys, encoded_vals)`` where

            * ``keys`` (list[str]) – ordered list of parameter names.
            * ``encoded_vals`` (numpy.ndarray) – logit-encoded initial values
              in the unconstrained search space.

        When ``ranges`` is ``None``, only the encoded values are returned (the
        keys are implicitly those of ``value_dict``).
    """
    if ranges is None:
        initial_vals = np.array([v for v in value_dict.values()])
        return _inverse_sigmoid(initial_vals)

    def scale(x, lower, upper):
        return (x - lower) / (upper - lower)

    res_keys = []
    res_vals = []
    for k, v in value_dict.items():
        if k not in ranges:
            res_vals.append(v)
        else:
            low, up = ranges[k]
            res_vals.append(scale(v, low, up))
        res_keys.append(k)

    # keys only in ranges
    for k in ranges.keys():
        if k not in value_dict:
            res_keys.append(k)
            res_vals.append(0.5)

    return res_keys, _inverse_sigmoid(np.array(res_vals))


def _compile_individual(x, param_keys=None, param_ranges=None, with_keys=True):
    """Decode an individual from the CMA-ES search space to human-readable values.

    Applies the sigmoid function and, where ``param_ranges`` are provided,
    linearly rescales from ``[0, 1]`` back to the original parameter range
    ``[lower, upper]``.

    Args:
        x (numpy.ndarray): 1-D array of parameter values in the internal
            (logit-encoded) search space.
        param_keys (list[str] or None): Ordered list of parameter names
            corresponding to the elements of ``x``.
        param_ranges (dict or None): Optional mapping of parameter name to
            ``(lower, upper)`` bounds used for rescaling.  Parameters not
            listed here are decoded with sigmoid only (result in ``(0, 1)``).
        with_keys (bool): If ``True`` (default), return a ``dict`` mapping
            parameter names to decoded values.  If ``False``, return a plain
            list of decoded values.

    Returns:
        dict or list: Decoded parameter values.  Type depends on ``with_keys``.
    """
    def scale_back(val, key):
        val = scipy.special.expit(val)
        low, up = param_ranges[key]
        return val * (up - low) + low

    # optional rescaling
    if param_ranges is None:
        ind = scipy.special.expit(x).tolist()
    else:
        ind = []
        for key, val in zip(param_keys, x):
            ind.append(val if key not in param_ranges else scale_back(val, key))

    # return with keys or not
    return _keys_with_evolved_vals(ind, param_keys) if with_keys else ind


def cma_es(model_func, hyperparam_config: dict, return_only_best=False, output_file=None, n_jobs=1):
    """Optimise model hyperparameters using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    Uses the ``cma`` library to iteratively propose candidate parameter
    vectors, evaluate them in parallel via ``EvalParallel2``, and update the
    evolution strategy.  An elitism mechanism replaces the worst individual of
    each generation with the all-time best solution found so far.

    Parameters are encoded in an unconstrained search space using the logit
    transform (see ``_init_values`` / ``_compile_individual``).

    Args:
        model_func (callable): Function with signature
            ``model_func(hyperparams: dict) -> dict`` to be minimised.
        hyperparam_config (dict): CMA-ES configuration dictionary.  Expected
            keys:

            * ``"MODEL"`` – initial parameter values (dict).
            * ``"SIGMA"`` – initial step size (float).
            * ``"CMA"`` – keyword arguments forwarded directly to
              ``cma.CMAEvolutionStrategy``.
            * ``"param_ranges"`` (optional) – per-parameter ``(lower, upper)``
              bounds for rescaling.

        return_only_best (bool): If ``True``, return only the best solution and
            its fitness without the full CMA-ES result object.  Defaults to
            ``False``.
        output_file (str or None): Path to a CSV log file.  Each individual
            evaluated is appended as one row.  Pass ``None`` to disable.
        n_jobs (int): Number of parallel worker processes for fitness
            evaluation.  Defaults to ``1``.

    Returns:
        dict: Result dictionary with the following keys:

            * ``"hyperparams"`` – decoded best parameter dictionary.
            * ``"result"`` – best fitness value found.
            * ``"es_data"`` (only when ``return_only_best=False``) – remaining
              fields from ``cma.CMAEvolutionStrategy.result``.
    """
    initial_kwargs = hyperparam_config["MODEL"]

    param_ranges = hyperparam_config.get("param_ranges", None)
    initial_keys, initial_vals = _init_values(initial_kwargs, ranges=param_ranges)

    sigma = hyperparam_config["SIGMA"]
    cma_kwargs = hyperparam_config["CMA"]

    _init_output_file(output_file, initial_keys)
    eval_func = partial(evaluate_with_params, model_func=model_func,
                        param_keys=initial_keys, param_ranges=param_ranges)

    es = cma.CMAEvolutionStrategy(initial_vals, sigma, cma_kwargs)

    gen_n = 0
    best_x = None
    best_f = float('inf')

    while not es.stop():
        X = es.ask()

        print("NJOBS", n_jobs)
        with EvalParallel2(fitness_function=eval_func, number_of_processes=n_jobs) as eval_all:
            fitnesses = eval_all(X)

        print("POPSIZE", len(fitnesses))
        for x, f in zip(X, fitnesses):
            if f < best_f:
                best_f = f
                best_x = x.copy()    

        # Replace worst with elite
        worst_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        X[worst_idx] = best_x.copy()
        fitnesses[worst_idx] = best_f

        es.tell(X, fitnesses)
        es.disp()

        for x, f in zip(X, fitnesses):
            ind = _compile_individual(x, param_keys=initial_keys,
                                      param_ranges=param_ranges, with_keys=False)
            _log_inidividual(output_file, list(ind), f, gen_n)

        gen_n += 1
        gc.collect()

    res = es.result
    x = _compile_individual(res[0], initial_keys, param_ranges=param_ranges)

    if return_only_best:
        return {"hyperparams": x, "result": res[1]}  # best evaluated solution, its objective function value
    return {"hyperparams": x, "result": res[1], "es_data": res[2:]}  # full result


hyperparam_search_zoo = {
    'GridSearch': perform_gridsearch,
    'CMA-ES': cma_es
}
