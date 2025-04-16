"""Plots scatter plots from UA analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FONTSIZE = 12

def plot_scatter(data: list[dict[str, str | pd.DataFrame]]) -> tuple[plt.figure, plt.axes]:
    """Plots scatter plot of all model runs."""

    num_plots = len(data)

    fig, axs = plt.subplots(figsize=(12, (3 * num_plots)), nrows=num_plots, ncols=1)

    for i, data_per_row in enumerate(data):
        
        xlabel = data_per_row["xlabel"]
        ylabel = data_per_row["ylabel"]
        df = data_per_row["data"]

        axis = axs[i] if num_plots > 1 else axs

        sns.scatterplot(df, x="xvalue", y="yvalue", hue="hue", ax=axis)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.legend().set_title("")
        axis.tick_params(axis="x", labelsize=FONTSIZE, rotation=0)
        axis.tick_params(axis="y", labelsize=FONTSIZE, rotation=0)
        
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
        name = "service_capacity"

    results = pd.read_csv(results_f, index_col=0)
    res_name = results[results["plot"] == name]

    assert not res_name.empty

    data = []

    for group in res_name.group.unique():
        plot_data = {}
        res_name_group = res_name[res_name.group == group]
        xaxis_label = res_name_group["xhue"].dropna().unique()[0]
        
        xaxis = res_name_group["xaxis"].unique().tolist()
        yaxis = res_name_group["yaxis"].unique().tolist()
        yaxis_label = res_name_group["ylabel"].unique()[0]
        assert len(yaxis) == 1
        yaxis_data = pd.read_csv(Path(root_dir, f"{yaxis[0]}.csv"), index_col="run")
        yaxis_data.columns = ["yvalue"]

        if yaxis[0] == "marginal_cost_carbon":
            yaxis_data.yvalue *= -1  # just for nicer plotting

        q01 = yaxis_data.quantile(0.01)
        q99 = yaxis_data.quantile(0.99)

        dfs = []
        for xcsv in xaxis:
            temp = pd.read_csv(Path(root_dir, f"{xcsv}.csv"), index_col="run")
            temp.columns = ["xvalue"]
            temp["hue"] = xcsv
            temp = pd.concat([temp, yaxis_data], axis=1)
            dfs.append(temp)
            
        df = pd.concat(dfs)

        # drop outliers

        df["to_plot"] = (df.yvalue <= q99.values[0]) & (df.yvalue >= q01.values[0])
        df = df[df.to_plot].drop(columns="to_plot")

        df["hue"] = df.hue.map(res_name_group.set_index("xaxis")["xlabel"])
        
        plot_data["xlabel"] = xaxis_label
        plot_data["ylabel"] = yaxis_label
        plot_data["data"] = df

        data.append(plot_data)

    fig, ax = plot_scatter(data)
    
    fig.savefig(out_f)