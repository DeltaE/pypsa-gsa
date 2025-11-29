"""Plots capacity results from UA"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FONTSIZE = 12


def plot_barplot(
    data: pd.DataFrame, xlabel: str | None = None, ylabel: str | None = None
) -> tuple[plt.figure, plt.axes]:
    """Plots scatter plot of all model runs."""

    df = data.copy().reset_index(drop=True)
    df = df.melt(var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=1)

    sns.barplot(df, x="metric", y="value", ax=ax)
    ax.set_xlabel(xlabel) if xlabel else ax.set_xlabel("")
    ax.set_ylabel(ylabel) if ylabel else ax.set_xlabel("")

    ax.set_xticks(range(len(df["metric"].unique())))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE, rotation=0)

    fig.tight_layout()

    return fig, ax


def retrieve_data(names: list[str], root_dir: str | Path) -> pd.DataFrame:
    """Assembels required results into a dataframe"""

    dfs = []

    for name in names:
        df = pd.read_csv(Path(root_dir, f"{name}.csv"), index_col="run")
        df = df.squeeze().to_frame(name=name)
        dfs.append(df)

    return pd.concat(dfs, axis=1)


if __name__ == "__main__":
    if "snakemake" in globals():
        root_dir = snakemake.params.root_dir
        results_f = snakemake.input.results
        out_f = snakemake.output.plot
        name = snakemake.wildcards.plot
    else:
        root_dir = "results/updates/ua/results/"
        results_f = "results/updates/ua/results.csv"
        out_f = "results/updates/ua/barplots/pwr_opt.png"
        name = "pwr_opt"

    results = pd.read_csv(results_f, index_col=0)
    results_filtered = results[results["ua_plot"] == name]
    assert len(results_filtered) > 0
    plot_names = results_filtered.index.to_list()

    df = retrieve_data(plot_names, root_dir)
    df = df.rename(columns=results_filtered["nice_name"])

    units = results_filtered.unit.values[
        0
    ]  # sanitize checks that all units are the same

    fig, ax = plot_barplot(df, ylabel=units)

    fig.savefig(out_f)
