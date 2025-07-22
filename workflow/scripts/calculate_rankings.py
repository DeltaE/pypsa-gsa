"""Prints out the union of the most impactful parameters."""

import pandas as pd

import logging

logger = logging.getLogger(__name__)


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
        top_n.extend(df[col].sort_values(ascending=True).index[:num_params].to_list())
    return sorted(list(set(top_n)))


def rank_params(df: pd.DataFrame) -> pd.DataFrame:
    """Rank the parameters."""
    na_columns = df.columns[df.isna().any()]
    if not na_columns.empty:
        logger.error(f"nans exist in the following columns: {na_columns}")
        raise ValueError(f"nans exist in the following columns: {na_columns}")

    df = df.dropna(axis=1)
    return df.rank(ascending=False, method="dense").astype(int)


if __name__ == "__main__":
    if "snakemake" in globals():
        input_f = snakemake.input.results
        rankings_f = snakemake.output.rankings_f
        top_n_f = snakemake.output.top_n_f
        top_n = snakemake.params.top_n
        subset = snakemake.params.subset
    else:
        input_f = "results/caiso/gsa/SA/all.csv"
        rankings_f = "results/caiso/gsa/rankings.csv"
        top_n_f = "results/caiso/gsa/top_n.csv"
        top_n = 3
        subset = [
            "objective_cost",
            "marginal_cost_energy",
            "marginal_cost_elec",
        ]

    df = pd.read_csv(input_f, index_col="param")
    ranked = rank_params(df)
    top_n_data = get_top_n_params(ranked, top_n, subset)

    ranked.to_csv(rankings_f, index=True)
    pd.DataFrame(top_n_data, columns=["param"]).to_csv(top_n_f, index=False)
