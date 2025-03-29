"""Plots scatter plots from UA analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any

def plot_scatter(data: list[dict[str, Any]]) -> tuple[plt.figure, plt.axes]:
    """Plots scatter plot of all model runs."""

    num_plots = len(data)

    fig, axs = plt.subplots(figsize=(12, (3 * num_plots)), nrows=num_plots, ncols=1)

    for i, ax_data in enumerate(data):
        xdata = ax_data["xaxis"]
        ydata = ax_data["yaxis"]

        df = pd.concat([xdata, ydata], axis=1)
        xlabel = xdata.columns.values[0]
        ylabel = ydata.columns.values[0]

        if num_plots > 1:
            sns.scatterplot(df, x=xlabel, y=ylabel, ax=axs[i])
        else:
            sns.scatterplot(df, x=xlabel, y=ylabel, ax=axs)

    fig.tight_layout()

    return fig, axs

if __name__ == "__main__":
    
    if "snakemake" in globals():
        root_dir = snakemake.params.root_dir
        results_f = snakemake.input.results
        out_f = snakemake.output.plot
        name = snakemake.wildcards.plot
    else:
        root_dir = "results/updates/ua/results/"
        results_f = "results/updates/ua/plots.csv"
        out_f = "results/updates/ua/plots/heat_pump.png"
        name = "heat_pump"

    results = pd.read_csv(results_f, index_col=0)
    results_filtered = results[results["plot"] == name]

    assert not results_filtered.empty

    data = []

    for name, row in results_filtered.iterrows():
        row_data = {}
        xaxis_f = Path(root_dir, f"{row['xaxis']}.csv")
        yaxis_f = Path(root_dir, f"{row['yaxis']}.csv")
        row_data["title"] = row["nice_name"]
        row_data["xlabel"] = row["xlabel"] if row["xlabel"] else row["xaxis"]
        row_data["ylabel"] = row["ylabel"] if row["ylabel"] else row["yaxis"]
        row_data["xaxis"] = (
            pd.read_csv(xaxis_f, index_col="run")
            .squeeze()
            .to_frame(name=row_data["xlabel"])
        )
        row_data["yaxis"] = (
            pd.read_csv(yaxis_f, index_col="run")
            .squeeze()
            .to_frame(name=row_data["ylabel"])
        )
        data.append(row_data)

    fig, ax = plot_scatter(data)
    
    fig.savefig(out_f)