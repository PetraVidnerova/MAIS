"""Command-line script for running many parallel MAIS epidemic simulations.

This script is the parallel, multi-repetition counterpart of
``run_experiment.py``.  It uses a ``Pool`` of worker processes to run
``n_repeat`` independent stochastic replicates of the same model and writes
the results either into a single ``.zip`` archive (one CSV per replicate,
default) or into a combined ``.feather`` file.

Typical usage::

    python run_multi_experiment.py config.ini my_run --n_repeat 100 --n_jobs 8

Output files are written to the directory specified by ``output_dir`` in the
config's ``[TASK]`` section.
"""

import sys
from zipfile import ZipFile
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
import click
import time
import timeit
import io
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

from utils.config_utils import ConfigFile, ConfigFileGenerator
from graphs.graph_gen import GraphGenerator
from models.model_zoo import model_zoo
from model_m.model_m import ModelM, load_model_from_config, load_graph
from utils.pool import Pool


def evaluate_model(model, setup):
    """Run one replicate of the model and return its results.

    This function is designed to be called by a worker process inside
    ``utils.pool.Pool``.  It resets the model to a new random seed, runs
    the simulation, and returns the resulting DataFrame alongside bookkeeping
    information.

    On ``AssertionError``, the failure is logged to a ``.FAILED`` file and
    ``(idx, None, None, None, None)`` is returned so the pool can continue.

    Args:
        model (ModelM): The model instance assigned to this worker.
        setup (tuple): A five-element tuple
            ``(idx, random_seed, test_id, config, args)`` where

            * ``idx`` (int) – worker index used to route answers back to the
              pool.
            * ``random_seed`` (int or None) – seed to pass to ``model.reset``.
              ``None`` means the seed is unchanged.
            * ``test_id`` (str) – tag appended to output file names.
            * ``config`` (ConfigFile) – configuration object (used only for
              error reporting).
            * ``args`` (tuple) – ``(ndays, print_interval, verbose)`` run
              parameters.

    Returns:
        tuple: ``(idx, df, deads, random_seed, suffix)`` where

        * ``idx`` (int) – worker index.
        * ``df`` (pandas.DataFrame or None) – per-day state history with an
          added ``"id"`` column.
        * ``deads`` – always ``None`` in the current implementation.
        * ``random_seed`` (int) – actual seed used by the model.
        * ``suffix`` (str) – file-name suffix derived from ``test_id``.
    """
    my_model = model

    idx, random_seed, test_id, config, args = setup
    ndays, print_interval, verbose = args

    suffix = "" if not test_id else "_" + test_id

    if random_seed is not None:
        my_model.reset(random_seed=random_seed)

    try:
        my_model.run(ndays, print_interval=print_interval, verbose=verbose)
    except AssertionError:
        file_name = f"history{suffix}.FAILED"
        if output_dir is not None:
            file_name = os.path.join(output_dir, file_name)
        with open(file_name, "w") as f:
            import sys
            import traceback
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            f.write(f'An error occurred on line {line} (file: {filename}) in statement {text}')
        return idx, None, None, None, None

    # # save history
    # file_name = f"history{suffix}.csv"
    # config.save(file_name)
    # cfg_string = ""
    # with open(file_name, "r") as f:
    #     cfg_string = "#" + "#".join(f.readlines())
    # with open(file_name, "w") as f:
    #     f.write(cfg_string)
    #     f.write(f"# RANDOM_SEED = {my_model.model.random_seed}\n")
    #     my_model.save_history(f)

    df = my_model.get_df()
    df["id"] = test_id
    random_seed = my_model.model.random_seed

    # with open(f"durations{suffix}.csv", "w") as f:
    #    my_model.model.save_durations(f)
    # with open(f"infect_time{suffix}.csv", "w") as f:
    #    for i in range(my_model.model.num_nodes):
    #        f.write(f"{my_model.model.infect_time[i]},")

    # save_source_infection = False
    # if save_source_infection:
    #     with open(f"sources{suffix}.csv", "w") as f:
    #         model.model.df_source_infection().to_csv(f)

    # save_dead = True
    # if save_dead:
    #     with open(f"dead{suffix}.csv", "w") as f:
    #         alld, young, old1, old2 = model.model.get_dead()
    #         print(f"{alld},{young},{old1},{old2}", file=f)

    if False:
        deads = pd.Series([idx, *model.model.get_dead()])
    else: 
        deads = None  

    return idx, df, deads, random_seed, suffix
#    del my_model


def demo(cf, test_id=None, model_random_seed=42,  print_interval=1, n_repeat=1, n_jobs=1,
         output_type="ZIP"):
    """Run many parallel epidemic model replicates and aggregate their output.

    Loads the graph once, creates ``n_jobs`` duplicates of the model, and
    feeds tasks into a ``Pool``.  Results arrive asynchronously and are written
    to disk on the fly (ZIP mode) or collected in memory and written at the
    end (FEATHER mode).

    Args:
        cf (ConfigFile): Loaded configuration object describing the experiment.
        test_id (str or None): Tag appended to output file and archive names.
            Defaults to ``None`` (no tag).
        model_random_seed (int or list[int]): Base seed (or explicit list of
            seeds for each replicate).  When an integer, seeds for replicates
            ``i`` are ``model_random_seed + i``.  Defaults to ``42``.
        print_interval (int): Simulated-day interval for progress output.
            Defaults to ``1``.
        n_repeat (int): Total number of replicates to run.  Defaults to ``1``.
        n_jobs (int): Number of parallel worker processes.  Defaults to ``1``.
        output_type (str): Output format – ``"ZIP"`` (default) saves each
            replicate as a separate CSV inside a zip archive; ``"FEATHER"``
            concatenates all results into a single feather file.

    Raises:
        ValueError: If ``output_type`` is not ``"ZIP"`` or ``"FEATHER"``.
    """
    #cf = ConfigFile()
    #cf.load(filename)
    graph = load_graph(cf)

    result_dfs = []
    result_series = []

    ndays = cf.section_as_dict("TASK").get("duration", 60)
    print_interval = cf.section_as_dict("TASK").get("print_interval", 1)
    verbose = cf.section_as_dict("TASK").get("verbose", "Yes") == "Yes"
    output_dir = cf.section_as_dict("TASK").get("output_dir", None)

    if output_type == "ZIP":
        suffix = f"_{test_id}" if test_id else ""
        zip_filename = os.path.join(output_dir, f"history{suffix}.zip")
        # prepare emtpy zip file
        with ZipFile(zip_filename, "w") as _:
            pass

    if not isinstance(model_random_seed, list):
        model_random_seed = [model_random_seed + i
                             for i in range(n_repeat)
                             ]

    # create model
    model = load_model_from_config(
        cf, model_random_seed[0], preloaded_graph=graph)
    print("model loaded", flush=True)

    # run parameters

    if test_id is None:
        test_id = ""

    models = [model]
    for i in range(1, n_jobs):
        models.append(model.duplicate(random_seed=model_random_seed[i]))

    pool = Pool(processors=n_jobs, evalfunc=evaluate_model, models=models)
    for i in range(n_jobs):
        pool.putQuerry((i, None, f"{test_id}_{i}", cf, (ndays, print_interval, verbose)))

    i = n_jobs
    answers = 0

    def _process_answer(idx, df, ser, random_seed, suffix):
        if output_type == "FEATHER":
            result_dfs.append(df)
            result_series.append(ser)
        elif output_type == "ZIP":
            save_to_zipped_csv(test_id,
                               idx,
                               df,
                               cf,
                               random_seed,
                               suffix,
                               zip_filename)
            result_series.append(ser)
        else:
            raise ValueError("unknown output_type option")
        print(f"Simulation {suffix if len(suffix) == 0 else suffix[1:]} finished.")

    while i < n_repeat:
        answer = pool.getAnswer()
        idx = answer[0]
        _process_answer(*answer)
        answers += 1
        pool.putQuerry((idx, model_random_seed[i],
                        f"{test_id}_{i}", cf, (ndays, print_interval, verbose)))
        i += 1

    for i in range(answers, n_repeat):
        _process_answer(*pool.getAnswer())

    pool.close()
    print("pool closed", flush=True)

    if output_type == "FEATHER":
        print("Concatenate results", flush=True)
        result_df = pd.concat(result_dfs).reset_index()
        print("Done.", flush=True)

        suffix = "" if not test_id else f"_{test_id}"
        file_name = f"history{suffix}.feather"
        if output_dir is not None:
            file_name = os.path.join(output_dir, file_name)
        result_df.to_feather(file_name)

    if False:
        result_df2 = pd.DataFrame(result_series)
        result_df2.columns = ["id", "all", "<65", "65-79", "80+"]

        file_name = f"deads_all{suffix}.csv"
        if output_dir is not None:
            file_name = os.path.join(output_dir, file_name)
        result_df2.to_csv(file_name)

    print("finished")


def save_to_zipped_csv(test_id, idx, df, config, random_seed, suffix, zipname):
    """Append a single replicate's result DataFrame to a shared ZIP archive.

    The CSV content is preceded by the configuration file text (each line
    prefixed with ``#``) and the random seed, matching the single-run CSV
    format produced by ``run_experiment.py``.

    Args:
        test_id (str): Experiment tag (used only for internal bookkeeping;
            the file name is derived from ``suffix``).
        idx (int): Replicate index (unused in the current implementation;
            reserved for future use).
        df (pandas.DataFrame): Per-day simulation results to write.
        config (ConfigFile): Configuration object whose string representation
            is prepended as comments.
        random_seed (int): Actual random seed used for this replicate.
        suffix (str): File-name suffix that forms part of the entry name
            inside the archive (e.g. ``"_my_run_3"``).
        zipname (str): Path to the ZIP archive to append to.  The archive
            must already exist (created by the caller).
    """
    file_name = f"history{suffix}.csv"

    cfg_string = config.to_string()
    # prepand all lines with #
    cfg_string = "\n".join(
        map(
            lambda x: "#" + x,
            cfg_string.split("\n")
        )
    )
    cfg_string += f"\n# RANDOM_SEED = {random_seed}\n"

    with io.StringIO() as df_str:
        df.to_csv(df_str)
        model_string = df_str.getvalue()

    output_string = cfg_string + model_string
    with ZipFile(zipname, "a") as zf:
        zf.writestr(file_name, output_string)


@click.command()
@click.option('--const-random-seed/--no-random-seed', ' /-r', default=True)
@click.option('--user-random-seeds', '-R', default=None)
@click.option('--print_interval',  default=1)
@click.option('--n_repeat',  default=1)
@click.option('--n_jobs', default=1)
@click.option('--log_level', default="CRITICAL", help="Logging level.")
@click.option('--output_type', default="ZIP", help="ZIP or FEATHER. ZIP produces `n_repeat` output csv files in one zip archive. FEATHER produces one big data frame in feather file.")
@click.argument('filename', default="example.ini")
@click.argument('test_id', default="")
def test(const_random_seed,  user_random_seeds,  print_interval, n_repeat, n_jobs, log_level, output_type,
         filename, test_id):
    """ Run the demo test inside the timeit """

    random_seed = 42 if const_random_seed else random.randint(0, 10000)
    if user_random_seeds is not None:
        with open(user_random_seeds, "r") as f:
            random_seeds = []
            for line in f:
                random_seeds.append(int(line.strip()))
            random_seed = random_seeds

    logging.basicConfig(format='%(levelname)s:%(module)s:%(lineno)d: %(message)s',
                        level=log_level)

    cf_generator = ConfigFileGenerator().load(filename)
    for cf in cf_generator:
        my_id = test_id
        suffix = cf.section_as_dict("OUTPUT_ID").get("id", None)    
        print(suffix)
        if suffix is not None:
            my_id += suffix


        print(f"Running {test_id}")

        def demo_fce(): return demo(cf, my_id,
                                    model_random_seed=random_seed,  print_interval=print_interval,
                                    n_repeat=n_repeat, n_jobs=n_jobs,
                                    output_type=output_type)
        print(timeit.timeit(demo_fce, number=1))


if __name__ == "__main__":
    test()
