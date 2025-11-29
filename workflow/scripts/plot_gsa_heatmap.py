from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging

logger = logging.getLogger(__name__)

FONTSIZE = 22


def plot_heatmap(df: pd.DataFrame) -> tuple[plt.figure, plt.axes]:
    fig, ax = plt.subplots(1, figsize=(12, 12))
    sns.heatmap(df, cmap="crest", ax=ax, cbar_kws={"label": "Scaled EE"})

    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE, rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONTSIZE)
    ax.figure.axes[-1].yaxis.label.set_size(FONTSIZE)

    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":
    if "snakemake" in globals():
        csvs = snakemake.input.csvs
        heatmap = snakemake.output.heatmap
        group = snakemake.wildcards.plot
        parameters_f = snakemake.input.params
        results_f = snakemake.input.results
    else:
        csvs = [
            "results/caiso/gsa/SA/objective_cost.csv",
            "results/caiso/gsa/SA/marginal_cost_energy.csv",
            "results/caiso/gsa/SA/marginal_cost_elec.csv",
            "results/caiso/gsa/SA/marginal_cost_carbon.csv",
        ]
        heatmap = "results/Testing/gsa/heatmaps/summary.png"
        group = ""
        parameters_f = "results/caiso/gsa/parameters.csv"
        results_f = "results/caiso/gsa/results.csv"

    dfs = []

    results = pd.read_csv(results_f, index_col=0).dropna(subset=["gsa_plot"]).copy()
    results = results[results.gsa_plot.str.contains(group)]
    r_nice_name = results.nice_name.to_dict()

    parameters = pd.read_csv(parameters_f)
    parameters = parameters.drop_duplicates(subset="group")
    p_nice_name = parameters.set_index("group")["nice_name"].to_dict()

    for csv in csvs:
        name = Path(csv).stem
        dfs.append(pd.read_csv(csv, index_col=0)["mu_star"].to_frame(r_nice_name[name]))

    if dfs:
        df = pd.concat(dfs, axis=1)
    else:
        df = pd.DataFrame(index=p_nice_name.values())

    df.index = df.index.map(p_nice_name)

    fig, ax = plot_heatmap(df)

    fig.savefig(heatmap)
