"""Plots scatter plots from UA analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatter(x_data: pd.Series, y_data: pd.Series, title: str | None = None) -> tuple[plt.figure, plt.axes]:
    """Plots scatter plot of all model runs."""

    xlabel = x_data.columns.values[0]
    ylabel = y_data.columns.values[0]
    
    title = title if title else ""

    df = pd.concat([x_data, y_data], axis=1)
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    sns.scatterplot(df, x=xlabel, y=ylabel, ax=ax)
    
    fig.tight_layout()
    
    return fig, ax

if __name__ == "__main__":
    
    if "snakemake" in globals():
        xaxis_f = snakemake.input.csvs[0]
        yaxis_f = snakemake.input.csvs[1]
        results_f = snakemake.input.results
        out_f = snakemake.output.plot
        name = snakemake.wildcards.plot
    else:
        xaxis_f = "results/updates/ua/results/hp_capacity_new.csv"
        yaxis_f = "results/updates/ua/results/marginal_cost_energy.csv"
        results_f = "results/updates/ua/plots.csv"
        out_f = "results/updates/ua/plots/hp_energy_cost.png"
        name = "hp_energy_cost"
        
    xaxis = pd.read_csv(xaxis_f, index_col="run")
    yaxis = pd.read_csv(yaxis_f, index_col="run")
    results = pd.read_csv(results_f, index_col=0)
    
    assert xaxis.shape[1] == 1
    assert yaxis.shape[1] == 1
    
    title = results.at[name, "nice_name"]
    xlabel = results.at[name, "xlabel"]
    ylabel = results.at[name, "ylabel"]
    
    if not xlabel:
        xlabel = results.at[name, "xaxis"] # non-verbose name
    if not ylabel:
        ylabel = results.at[name, "yaxis"] # non-verbose name
    
    xaxis = xaxis.squeeze().to_frame(name=xlabel)
    yaxis = yaxis.squeeze().to_frame(name=ylabel)
    
    fig, ax = plot_scatter(xaxis, yaxis, title)
    
    fig.savefig(out_f)