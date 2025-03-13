from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df: pd.DataFrame) -> tuple[plt.figure, plt.axes]:
    fig, ax = plt.subplots(1, figsize=(12,12))
    sns.heatmap(df, cmap="crest", ax=ax)
    
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=45, ha="right")
    
    # ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    
    return fig, ax

if __name__ == "__main__":
    if "snakemake" in globals():
        csvs = snakemake.input.csvs
        heatmap = snakemake.output.heatmap
        group = snakemake.wildcards.group
    else:
        csvs = [
            "results/Testing/SA/com_marginal_cost_elec.csv",
            "results/Testing/SA/com_marginal_cost_energy.csv",
            "results/Testing/SA/res_marginal_cost_elec.csv",
            "results/Testing/SA/res_marginal_cost_energy.csv",
            "results/Testing/SA/ind_marginal_cost_elec.csv",
            "results/Testing/SA/ind_marginal_cost_energy.csv",
            "results/Testing/SA/trn_marginal_cost_elec.csv",
            "results/Testing/SA/trn_marginal_cost_energy.csv",
            "results/Testing/SA/marginal_cost_elec.csv",
            "results/Testing/SA/marginal_cost_energy.csv",
            # "results/Testing/SA/objective_cost.csv"
        ]
        heatmap = "results/Testing/heatmaps/marginal_cost.png"
        group = ""

    dfs = []
    
    for csv in csvs:
        name = Path(csv).stem
        dfs.append(pd.read_csv(csv, index_col=0)["mu_star"].to_frame(name))
        
    df = pd.concat(dfs, axis=1)
    fig, ax = plot_heatmap(df)
    
    fig.savefig(heatmap)
