from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger(__name__)

FONTSIZE = 22

def plot_barchart(df: pd.DataFrame) -> tuple[plt.figure, plt.axes]:
    """Plots normalized data."""
    
    df = normalize(df)
    df = format(df)
    
    fig, ax = plt.subplots(1, figsize=(12,12))
    
    sns.barplot(data=df, ax=ax, x="Value", y="Group", hue="Result", orient="h", errorbar=None)
    
    ax.tick_params(labelsize=FONTSIZE)
    ax.set_ylabel("")
    ax.set_xlabel("μ/μ*", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)

    fig.tight_layout()
    
    return fig, ax

def format(results: pd.DataFrame) -> pd.DataFrame:
    """Formats data for plotting"""
    df = results.copy()
    return df.reset_index(names="Group").melt(id_vars=["Group"], var_name="Result", value_name="Value").sort_values(by=["Result","Value"])

def normalize(results: pd.DataFrame) -> pd.DataFrame:
    """Normalizes all data to be between 0 and 1."""
    
    df = results.copy()
    
    for column in df.columns:
        max_value = df[column].max()
        df[column] = df[column].div(max_value)
        
    return df

if __name__ == "__main__":
    if "snakemake" in globals():
        csvs = snakemake.input.csvs
        barplot = snakemake.output.barplot
        group = snakemake.wildcards.group
        parameters_f = snakemake.input.params
        results_f = snakemake.input.results
    else:
        csvs = [
            "results/Testing/gsa/SA/objective_cost.csv",
            "results/Testing/gsa/SA/marginal_cost_energy.csv",
            "results/Testing/gsa/SA/marginal_cost_elec.csv",
            "results/Testing/gsa/SA/marginal_cost_carbon.csv"
        ]
        barplot = "results/Testing/barplots/summary.png"
        group = ""
        parameters_f = "results/Testing/parameters.csv"
        results_f = "results/Testing/results.csv"

    dfs = []

    results = pd.read_csv(results_f, index_col=0)
    results = results[results.plots.str.contains(group)]
    r_nice_name = results.nice_name.to_dict()

    parameters = pd.read_csv(parameters_f)
    parameters = parameters.drop_duplicates(subset="group")
    p_nice_name = parameters.set_index("group")["nice_name"].to_dict()

    for csv in csvs:
        name = Path(csv).stem
        dfs.append(pd.read_csv(csv, index_col=0)["mu_star"].to_frame(r_nice_name[name]))
    df = pd.concat(dfs, axis=1)

    df.index = df.index.map(p_nice_name)

    fig, ax = plot_barchart(df)
    
    fig.savefig(barplot)
