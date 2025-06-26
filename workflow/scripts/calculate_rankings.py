"""Prints out the union of the most impactful parameters."""

import pandas as pd


def get_top_n_params(
    raw: pd.DataFrame, num_params: int, results: list[str]
) -> list[str]:
    """Get the top n most impactful parameters."""

    df = raw.copy()[results]

    if df.empty:
        print("No top_n parameters found for GSA")
        return []

    top_n = []
    for col in df.columns:
        top_n.extend(df[col].sort_values(ascending=False).index[:num_params].to_list())
    return sorted(list(set(top_n)))

def rank_params(df: pd.DataFrame) -> pd.DataFrame:
    """Rank the parameters."""
    return df.rank(ascending=False, method="dense")

if __name__ == "__main__":
    if "snakemake" in globals():
        input_f = snakemake.input.csvs
        rankings_f = snakemake.output.rankings
        top_n = snakemake.params.top_n
        subset = snakemake.input.subset
        top_n_f = snakemake.output.top_n
    else:
        input_f = "results/gsa/Testing/results.csv"
        rankings_f = "results/gsa/Testing/rankings.csv"
        top_n = 5
        subset = ["objective_cost", "marginal_cost_energy", "marginal_cost_elec", "marginal_cost_carbon"]
        top_n_f = "results/gsa/Testing/top_n.csv"
        
    df = pd.read_csv(input_f, index_col=0)
    ranked = rank_params(df)
    top_n_data = get_top_n_params(ranked, top_n, subset)

    df.index = df.index.map(p_nice_name)