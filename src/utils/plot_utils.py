"""Plotting utilities for MAIS simulation histories.

This module provides functions for loading simulation output CSV files and
visualising epidemic curves and other time-series metrics via Matplotlib and
Seaborn. It supports single-run plots, multi-run aggregated line plots, and
animated state-histogram views.

Public API:
    - :func:`plot_history`: Quick plot of a single history file.
    - :func:`plot_histories`: Aggregate multiple runs on one axis.
    - :func:`plot_mutliple_policies`: Compare policies on a single metric.
    - :func:`plot_mutliple_policies_everything`: Multi-panel comparison across
      all tracked metrics.
    - :func:`plot_state_histogram`: Animated per-state bar chart over time.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import animation
from typing import Dict, List


def plot_history(filename: str):
    """Plot the ``all_infectious`` curve from a single simulation history CSV.

    Loads the history file, plots ``all_infectious`` against ``T`` using the
    default Pandas/Matplotlib backend, and displays the figure interactively.

    Args:
        filename (str): Path to the simulation output CSV file.
    """
    history = _load_history(filename)
    history.plot(x="T", y="all_infectious")
    plt.show()


def plot_histories(*args, group_days: int = None, group_func: str = "max", **kwargs):
    """Plot ``all_infectious`` from multiple simulation history CSV files.

    Loads each history file, optionally groups records into day-buckets, then
    overlays all runs on a single line plot using :func:`_plot_lineplot`.

    Args:
        *args (str): One or more paths to simulation output CSV files.
        group_days (int, optional): If set, aggregates rows into buckets of
            this many days before plotting. Defaults to ``None`` (no
            grouping).
        group_func (str, optional): Aggregation function applied within each
            day bucket (e.g., ``"max"``, ``"mean"``). Defaults to ``"max"``.
        **kwargs: Additional keyword arguments forwarded to
            :func:`_plot_lineplot` (e.g., ``title``, ``save_path``).
    """
    histories = [_history_with_fname(
        filename, group_days=group_days, group_func=group_func) for filename in args]
    history_one_df = pd.concat(histories)
    _plot_lineplot(history_one_df, "day", "all_infectious", **kwargs)


def plot_mutliple_policies(policy_dict: Dict[str, List[str]],
                           group_days: int = None, group_func: str = "max", value="all_infectious", max_days=None, **kwargs):
    """Compare a single metric across multiple policies on one line plot.

    For each policy, loads all associated history files, concatenates them,
    and renders a median line with inter-quartile shading using
    :func:`_plot_lineplot`.

    Args:
        policy_dict (Dict[str, List[str]]): Mapping of policy name to a list
            of history CSV file paths for that policy.
        group_days (int, optional): Day-bucket size for temporal aggregation.
            Defaults to ``None`` (no grouping).
        group_func (str, optional): Aggregation function applied per bucket.
            Defaults to ``"max"``.
        value (str, optional): Column name of the metric to plot on the
            y-axis. Defaults to ``"all_infectious"``.
        max_days (int, optional): If set, truncates each history to the first
            ``max_days`` rows. Defaults to ``None``.
        **kwargs: Additional keyword arguments forwarded to
            :func:`_plot_lineplot`.
    """
    histories = []
    for policy_key, history_list in policy_dict.items():
        histories.extend([_history_with_fname(filename,
                                              group_days=group_days,
                                              group_func=group_func,
                                              policy_name=policy_key,
                                              max_days=max_days)
                          for filename in history_list])

    history_one_df = pd.concat(histories)
    _plot_lineplot(history_one_df, "day", value,
                   hue="policy_name", **kwargs)


def plot_mutliple_policies_everything(policy_dict: Dict[str, List[str]],
                                      group_days: int = None, group_func: str = "max",
                                      max_days=None, **kwargs):
    """Render a multi-panel comparison of all tracked metrics across multiple policies.

    Loads and concatenates histories for every policy, then delegates to
    either :func:`_plot_lineplot2` (variant 2) or :func:`_plot_lineplot3`
    (default) depending on the optional ``variant`` keyword argument.

    Args:
        policy_dict (Dict[str, List[str]]): Mapping of policy name to a list
            of history CSV file paths for that policy.
        group_days (int, optional): Day-bucket size for temporal aggregation.
            Defaults to ``None`` (no grouping).
        group_func (str, optional): Aggregation function applied per bucket.
            Defaults to ``"max"``.
        max_days (int, optional): If set, truncates each history to the first
            ``max_days`` rows. Defaults to ``None``.
        **kwargs: Additional keyword arguments forwarded to the chosen plot
            function. The special key ``variant`` (int) selects the plot
            layout (``2`` selects :func:`_plot_lineplot2`; any other value
            selects :func:`_plot_lineplot3`). The ``title`` key is required
            by the underlying plot functions.
    """
    histories = []
    for policy_key, history_list in policy_dict.items():
        histories.extend([_history_with_fname(filename,
                                              group_days=group_days,
                                              group_func=group_func,
                                              policy_name=policy_key,
                                              max_days=max_days)
                          for filename in history_list])

    history_one_df = pd.concat(histories)

    if "variant" in kwargs and kwargs["variant"] == 2:
        plot_function = _plot_lineplot2
        del kwargs["variant"]
    else:
        plot_function = _plot_lineplot3

    plot_function(history_one_df, "day",
                  hue="policy_name", **kwargs)


def plot_state_histogram(filename: str, title: str = "Simulation", states: List[str] = None, save_path: str = None):
    """Render an animated bar chart showing the per-state population over time.

    Reads the given history CSV and produces an animation where each frame
    corresponds to one simulation day. Each bar represents a disease/model
    state, and its height equals the number of nodes in that state on the
    corresponding day.

    Args:
        filename (str): Path to the simulation output CSV file.
        title (str, optional): Base title displayed in the figure. The current
            day number is appended dynamically per frame. Defaults to
            ``"Simulation"``.
        states (List[str], optional): Subset of state column names to include
            in the histogram. If ``None``, all state columns present in the
            CSV (excluding metadata columns) are shown. Defaults to ``None``.
        save_path (str, optional): If provided, the animation is saved to this
            file path using FFMpeg at 10 fps before being displayed. Defaults
            to ``None``.
    """
    def animate(i):
        fig.suptitle(f"{title} - day {day_labels.iloc[i]}")

        data_i = data.iloc[i]
        for d, b in zip(data_i, bars):
            b.set_height(math.ceil(d))

    fig, ax = plt.subplots()

    history = _history_with_fname(filename, group_days=1, keep_only_all=False)
    day_labels = history["day"]
    data = history.drop(["T", "day", "all_infectious", "filename"], axis=1)
    if states is not None:
        data = data[states]

    bars = plt.barplot(range(data.shape[1]),
                       data.values.max(), tick_label=data.columns)

    anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=history.shape[0],
                                   interval=100)

    if save_path is not None:
        anim.save(save_path, writer=animation.FFMpegWriter(fps=10))
    plt.show()


def _plot_lineplot(history_df, x, y, hue=None, save_path=None, **kwargs):
    """Render a median line plot with IQR shading, grouped by a hue column.

    For each unique value in the ``hue`` column the function draws the median
    trajectory and fills between the 25th and 75th percentiles. Y-axis limits
    are hard-coded based on the metric name (``mean_waiting`` → ``[0, 10]``;
    everything else → ``[0, 150]``).

    Args:
        history_df (pandas.DataFrame): Combined history data for all policies/
            runs.
        x (str): Column name to use as the x-axis (typically ``"day"``).
        y (str): Column name to use as the y-axis metric.
        hue (str, optional): Column name used to separate runs into groups
            (e.g., ``"policy_name"``). Defaults to ``None``.
        save_path (str, optional): If provided, the figure is saved to this
            path before display. Defaults to ``None``.
        **kwargs: Additional keyword arguments. The ``title`` key (str) is
            extracted and applied as the axes title; remaining keys are
            currently unused.
    """
    if "title" in kwargs:
        title = kwargs["title"]
        del kwargs["title"]
    else:
        title = ""
    fig, axs = plt.subplots()

    for policy in history_df[hue].unique():
        policy_df = history_df[history_df[hue] == policy]

        policy_stats = policy_df.groupby([x, policy]).describe()
        q1 = policy_stats[y]["25%"]
        q3 = policy_stats[y]["75%"]

        sns.lineplot(x=x, y=y, data=policy_df, label=policy,
                     estimator=np.median, ci=None, ax=axs)
        axs.fill_between(np.arange(len(policy_df[x].unique())), q1, q3, alpha=0.3)

    # dirty hack (ro)
    if y == "mean_waiting":
        axs.set(ylim=(0, 10))
    else:
        axs.set(ylim=(0, 150))
    axs.set_title(title)
    if save_path is not None:
        fig.savefig(save_path)

    plt.show()


def _plot_lineplot2(history_df, x, hue=None, save_path=None, plotall=True, **kwargs):
    """Render a two-panel line plot comparing detected and total infectious cases.

    Produces a side-by-side figure with:
    - Left panel: median ``I_d`` (detected active cases).
    - Right panel: median ``all_infectious`` (all active cases, excluding the
      ``"Czech Republic (scaled down)"`` policy).

    Args:
        history_df (pandas.DataFrame): Combined history data for all policies/
            runs.
        x (str): Column name for the x-axis (typically ``"day"``).
        hue (str, optional): Column name used to colour lines by group.
            Defaults to ``None``.
        save_path (str, optional): If provided, the figure is saved to this
            path. Defaults to ``None``.
        plotall (bool, optional): Unused in this variant; kept for API
            consistency. Defaults to ``True``.
        **kwargs: Must contain ``title`` (str) for the figure super-title.
            Optional keys: ``maxy`` (int, y-axis upper limit) and ``maxx``
            (int, x-axis upper limit).
    """

    title = kwargs["title"]
    del kwargs["title"]
    maxy = kwargs.get("maxy", None)
    if "maxy" in kwargs:
        del kwargs["maxy"]

    maxx = kwargs.get("maxx", None)
    if "maxx" in kwargs:
        del kwargs["maxx"]

    fig = plt.figure()
    axs = [None] * 2
    axs[0] = fig.add_subplot(121)
    axs[1] = fig.add_subplot(122)
#    axs[2] = fig.add_subplot(223)
#    axs[3] = fig.add_subplot(224)

    # dirty hack to get rid of stupid legend title
    kwargs["legend"] = False
    sns_plot = sns.lineplot(x=x, y="I_d", data=history_df,
                            hue=hue, estimator=np.median, ci='sd', ax=axs[0], **kwargs)

    history_df_r = history_df[history_df["policy_name"]
                              != "Czech Republic (scaled down)"]
#    maxy = 5000
    # dirty hack (ro)
    axs[0].set(ylim=(0, maxy))
    axs[0].set(xlim=(0, maxx))
    axs[0].set_ylabel("all detected states")
    axs[0].set_title("detected - active cases - median")
    axs[0].legend(history_df[hue].unique(), title=None, fancybox=True, )

    sns_plot2 = sns.lineplot(x=x, y="all_infectious", data=history_df_r,
                             hue=hue, estimator=np.median, ci='sd', ax=axs[1], **kwargs)
#    maxy = 25000
    # dirty hack (ro)
    axs[1].set(ylim=(0, maxy))
    axs[1].set(xlim=(0, maxx))
    axs[1].set_ylabel("all infected states")
    axs[1].set_title("all active cases - median")
    axs[1].legend(history_df_r[hue].unique(), title=None, fancybox=True, )

    """
    sns_plot = sns.lineplot(x=x, y="I_d", data=history_df,
                            hue=hue, estimator=np.mean, ci='sd', ax=axs[2], **kwargs)

    history_df_r = history_df[history_df["policy_name"] != "Czech Republic (scaled down)"]

    maxy = 5000
    # dirty hack (ro)
    axs[2].set(ylim=(0, maxy))
    axs[2].set(xlim=(0, maxx))
    axs[2].set_ylabel("all detected states")
    axs[2].set_title("detected - active cases - mean")
    axs[2].legend(history_df[hue].unique(), title=None, fancybox=True, )

    sns_plot2 = sns.lineplot(x=x, y="all_infectious", data=history_df_r,
                             hue=hue, estimator=np.mean, ci='sd', ax=axs[3], **kwargs)
    maxy = 25000
    # dirty hack (ro)
    axs[3].set(ylim=(0, maxy))
    axs[3].set(xlim=(0, maxx))
    axs[3].set_ylabel("all infected states")
    axs[3].set_title("all active cases - mean")
    axs[3].legend(history_df_r[hue].unique(), title=None, fancybox=True, )
    """

    fig.suptitle(title, fontsize=20)

    if save_path is not None:
        plt.savefig(save_path)


def _plot_lineplot3(history_df, x, hue=None, save_path=None, plotall=True, **kwargs):
    """Render a multi-panel line plot covering all key epidemic metrics.

    Produces up to six panels depending on ``plotall``:
    - Panel 0: Median detected active cases (``I_d``).
    - Panel 1: Median total infectious cases (``all_infectious``).
    - Panel 2 (if ``plotall``): Median mean waiting time (``mean_waiting``).
    - Panel 3: Median detection ratio (``detected_ratio``).
    - Panel 4 (if ``plotall``): Median total tests (``all_tests``).
    - Panel 5 (if ``plotall``): Median mean infection probability
      (``mean_p_infection``).

    Vertical reference lines are drawn at days 5, 36, 66, and 97 on
    the panels that show them.

    Args:
        history_df (pandas.DataFrame): Combined history data for all policies/
            runs.
        x (str): Column name for the x-axis (typically ``"day"``).
        hue (str, optional): Column name used to colour lines by group.
            Defaults to ``None``.
        save_path (str, optional): If provided, the figure is saved to this
            path. Defaults to ``None``.
        plotall (bool, optional): Whether to include the additional three
            panels (waiting time, tests, infection probability). Defaults to
            ``True``.
        **kwargs: Must contain ``title`` (str) for the figure super-title.
            Optional key: ``maxy`` (int, y-axis upper limit; default ``300``).
    """

    title = kwargs["title"]
    del kwargs["title"]
    maxy = kwargs.get("maxy", 300)
    if "maxy" in kwargs:
        del kwargs["maxy"]

    fig = plt.figure()
    axs = [None] * 6
    axs[0] = fig.add_subplot(131)
    axs[1] = fig.add_subplot(132)
    if plotall:
        axs[2] = fig.add_subplot(433)
        axs[4] = fig.add_subplot(436)
        axs[3] = fig.add_subplot(439)
        axs[5] = fig.add_subplot(4, 3, 12)
    else:
        axs[3] = fig.add_subplot(133)

    # axs[2] = fig.add_subplot(433)
    # axs[3] = fig.add_subplot(436)
    # axs[4] = fig.add_subplot(439)
    # axs[5] = fig.add_subplot(4,3,12)

    # dirty hack to get rid of stupid legend title
    kwargs["legend"] = False
    sns_plot = sns.lineplot(x=x, y="I_d", data=history_df,
                            hue=hue, estimator=np.median, ci='sd', ax=axs[0], **kwargs)

    history_df_r = history_df[history_df["policy_name"]
                              != "Czech Republic (scaled down)"]

    # dirty hack (ro)
    axs[0].set(ylim=(0, 40))
#    axs[0].set(xlim=(1, 150))
    axs[0].set_ylabel("all detected states")
    axs[0].set_title("detected - active cases")
    axs[0].legend(history_df[hue].unique(), title=None, fancybox=True, )
    axs[0].axvline(x=5, color="gray")
    axs[0].axvline(x=36, color="gray")
    axs[0].axvline(x=66, color="gray")
    axs[0].axvline(x=97, color="gray")

    sns_plot2 = sns.lineplot(x=x, y="all_infectious", data=history_df_r,
                             hue=hue, estimator=np.median, ci='sd', ax=axs[1], **kwargs)
    # dirty hack (ro)
    axs[1].set(ylim=(0, 150))
 #   axs[1].set(xlim=(1, 150))
    axs[1].set_ylabel("all infected states")
    axs[1].set_title("all active cases")
    axs[1].legend(history_df_r[hue].unique(), title=None, fancybox=True, )

    if plotall:
        sns_plot3 = sns.lineplot(x=x, y="mean_waiting", data=history_df,
                                 hue=hue, estimator=np.median, ci='sd', ax=axs[2], **kwargs)
        axs[2].set(ylim=(0, 15))
        axs[2].set(xlim=(1, 120))
        axs[2].legend(history_df[hue].unique(), title=None, fancybox=True, )


#        axs[2].set_title("waiting times")

    sns.lineplot(x=x, y="detected_ratio", data=history_df_r,
                 hue=hue, estimator=np.median, ci=None, ax=axs[3], **kwargs)
    axs[3].set(ylim=(0, 15))
    axs[3].set(xlim=(1, 120))
    axs[3].legend(history_df_r[hue].unique(), title=None, fancybox=True, )

    #   axs[3].set_title("detected_ratio")
    axs[3].axvline(x=5)
    axs[3].axvline(x=36)
    axs[3].axvline(x=66)
    axs[3].axvline(x=97)
    axs[3].axhline(y=10)

    sns.lineplot(x=x, y="all_tests", data=history_df,
                 hue=hue, estimator=np.median, ci='sd', ax=axs[4],
                 **kwargs)
    axs[4].set(ylim=(0, 52))
    axs[4].set(xlim=(1, 120))
    axs[4].legend(history_df[hue].unique(), title=None, fancybox=True, )

    sns.lineplot(x=x, y="mean_p_infection", data=history_df_r,
                 hue=hue, estimator=np.median, ci='sd', ax=axs[5],
                 **kwargs)
    axs[5].set(xlim=(1, 120))
    axs[5].legend(history_df_r[hue].unique(), title=None, fancybox=True, )

    # sns.lineplot(x=x, y="nodes_in_quarantine", data=history_df,
    #              hue=hue, estimator=np.mean, ci='sd', ax=axs[3], **kwargs)
    # sns.lineplot(x=x, y="contacts_collected", data=history_df,
    #              hue=hue, estimator=np.mean, ci='sd', ax=axs[4], **kwargs)
    # sns.lineplot(x=x, y="released_nodes", data=history_df,
    #              hue=hue, estimator=np.mean, ci='sd', ax=axs[5], **kwargs)
    # axs[3].set(ylim=(0, 200))
    # axs[3].set_title("nodes_in_quarantines")

    # axs[4].set(ylim=(0, 50))
    # axs[4].set_title("contacts_collected")

    # axs[5].set(ylim=(0, 50))
    # axs[5].set_title("released_nodes")

    #    sns.lineplot(x=x, y="", data=history_df,
    #                 hue=hue, estimator=np.median, ci='sd', ax=axs[3], **kwargs)
    #    axs[3].set(ylim=(0, 15))
    #    axs[3].set_title("waiting times")

    """
    sns.lineplot(x=x, y="tests_ratio", data=history_df,
                 hue=hue, estimator=np.median, ci='sd', ax=axs[3], **kwargs)
    sns.lineplot(x=x, y="tests_ratio_to_s", data=history_df,
                 hue=hue, estimator=np.median, ci='sd', ax=axs[3], **kwargs)
    axs[3].set_title("tests ratio to all infected, symptomatic infected")
    axs[3].set(ylim=(0, None))
    """

    fig.suptitle(title, fontsize=20)

    if save_path is not None:
        plt.savefig(save_path)

#    plt.show()


def _history_with_fname(filename, group_days: int = None, group_func: str = "max", policy_name: str = None,
                        keep_only_all: bool = False, max_days=None):
    """Load a history CSV and enrich it with metadata columns.

    Reads the history file via :func:`_load_history`, optionally restricts
    columns to ``["day", "all_infectious"]``, applies temporal bucketing, and
    inserts ``filename`` (and optionally ``policy_name``) as new columns.

    Args:
        filename (str): Path to the simulation output CSV file.
        group_days (int, optional): If set to a positive integer, rows are
            bucketed into intervals of this many days and aggregated using
            ``group_func``. Defaults to ``None`` (no grouping).
        group_func (str, optional): Pandas aggregation function applied within
            each day bucket. Defaults to ``"max"``.
        policy_name (str, optional): If provided, a ``policy_name`` column is
            added to the resulting DataFrame. Defaults to ``None``.
        keep_only_all (bool, optional): If ``True``, all columns except
            ``"day"`` and ``"all_infectious"`` are dropped before grouping.
            Defaults to ``False``.
        max_days (int, optional): Maximum number of rows to retain (passed to
            :func:`_load_history`). Defaults to ``None``.

    Returns:
        pandas.DataFrame: Processed history DataFrame with at minimum the
        columns ``"filename"``, ``"day"``, and ``"all_infectious"``.
    """
    history = _load_history(filename, max_days=max_days)
    if keep_only_all:
        history = history[["day", "all_infectious"]]

    if group_days is not None and group_days > 0:
        history["day"] = history["day"] // group_days * group_days
        history = history.groupby(
            "day", as_index=False).agg(func=group_func)

    history.insert(0, "filename", filename)

    if policy_name is not None:
        history["policy_name"] = policy_name
    return history


def _load_history(filename: str, max_days=None) -> pd.DataFrame:
    """Load and pre-process a simulation history CSV into a DataFrame.

    Reads the CSV, computes derived aggregate columns (e.g.,
    ``all_infectious``, ``I_d``, ``all_tests``, ``detected_ratio``,
    ``tests_ratio``, ``mean_waiting``, ``nodes_in_quarantine``,
    ``released_nodes``, ``contacts_collected``), and ensures a ``"day"``
    column is always present.

    Derivations are performed only when an ``"E"`` column is present in the
    CSV (indicating a full SEIR-type output format). For simpler outputs,
    several columns are filled with zeros so downstream code can always
    reference them.

    Args:
        filename (str): Path to the simulation output CSV file. Lines
            starting with ``'#'`` are treated as comments and skipped.
        max_days (int, optional): If set, truncates the DataFrame to the
            first ``max_days`` rows after loading. Defaults to ``None``
            (no truncation).

    Returns:
        pandas.DataFrame: Pre-processed history with derived aggregate columns
        added. A ``"day"`` column is guaranteed to exist (falls back to a
        sequential integer range if not present in the source file).
    """
    print(filename)
    history = pd.read_csv(filename, comment="#")
    if "E" in history.columns:

        all_infectious = [
            s
            for s in [
                "I_n", "I_a", "I_s", "E",
                "I_dn", "I_da", "I_ds", "E_d", "J_ds", "J_dn",
                "J_n", "J_s"]
            if s in history.columns
        ]

        history["all_infectious"] = history[all_infectious].sum(axis=1)
        try:
            history["I_d"] = history[[
                "I_dn", "I_da", "I_ds", "E_d", "J_ds", "J_dn"]].sum(axis=1)
            history["all_tests"] = history[[
                "tests", "quarantine_tests"]].sum(axis=1)
                
            history["detected_ratio"] = history["all_infectious"] / history["I_d"]
            
            history["tests_ratio"] = history["tests"] / \
                                     history["all_infectious"]
                
            history["all_s"] = history[[
                "I_s", "I_ds", "J_s", "J_ds"]].sum(axis=1)
            history["tests_ratio_to_s"] = history["tests"] / history["all_s"]
                
            history["mean_waiting"] = history["sum_of_waiting"] / \
                                      history["all_positive_tests"]
                
            selected_cols = [
                col
                for col in history.columns
                if "nodes_in_quarantine" in col
            ]
            history["nodes_in_quarantine"] = history[selected_cols].sum(axis=1)
            selected_cols = [
                col
                for col in history.columns
                if "released_nodes" in col
            ]
            history["released_nodes"] = history[selected_cols].sum(axis=1)
            selected_cols = [
                col
                for col in history.columns
                if "contacts_collected" in col
            ]
            history["contacts_collected"] = history[selected_cols].sum(axis=1)
        except KeyError:
            print("Warning something is missing in data frame")
            

    else:
        history["nodes_in_quarantine"] = 0
        history["released_nodes"] = 0
        history["contacts_collected"] = 0
        history["mean_p_infection"] = 0

    if max_days is not None:
        history = history[:max_days]
    if "day" not in history.columns:
        history["day"] = range(len(history))
#    print(history)
    return history


if __name__ == "__main__":

    history = pd.read_csv(
        "../result_storage/tmp/history_seirsplus_quarantine_1.csv")
    plot_history(history)
