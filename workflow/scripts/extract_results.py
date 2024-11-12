"""Extracts results to have SA run over"""

import pandas as pd
import pypsa


def _get_p_nom_opt(links: pd.DataFrame, carriers: list[str]) -> float:
    return links[links.carrier.isin(carriers)].p_nom_opt.sum()


def _get_p_total(
    links: pd.DataFrame,
    links_t: pd.DataFrame,
    carriers: list[str],
    sns_weights: pd.Series,
) -> float:
    slicer = links[links.carrier.isin(carriers)].index
    df = links_t.copy()
    df = df[slicer]
    assert all(df <= 0)
    return df.mul(sns_weights, axis=0).mul(-1).sum().sum()


def _get_objective_cost(n: pypsa.Network) -> float:
    return n.objective


def _extract_carriers(cars: str | None) -> list[str] | None:

    if not cars:
        return None
    else:
        return [y.strip() for y in cars.split(";")]


def extract_results(n: pypsa.Network, results: pd.DataFrame) -> pd.DataFrame:

    res = results.copy().set_index("name")
    res["carriers"] = res.carriers.map(_extract_carriers)

    assert len(n.investment_periods) == 1
    year = n.investment_periods[0]

    links = n.links
    links_t_p1 = n.links_t["p1"].loc[year]
    sns_weights = n.snapshot_weightings.loc[year].objective

    data = []

    for name, row in res.iterrows():

        variable = row["variable"]
        carriers = row["carriers"]

        if variable == "p_nom_opt":
            value = _get_p_nom_opt(links, carriers)
        elif variable == "p_total":
            value = _get_p_total(links, links_t_p1, carriers, sns_weights)
        elif variable == "objective_cost":
            value = _get_objective_cost(n)

        data.append([name, value])

    return pd.DataFrame(data, columns=["name", "value"])


if __name__ == "__main__":
    if "snakemake" in globals():
        network = snakemake.input.network
        results_f = snakemake.params.results
        model_run = snakemake.wildcards.run
        csv = snakemake.output.csv
    else:
        network = "results/California/modelruns/40/network.nc"
        results_f = "config/results.csv"
        csv = "results/California/modelruns/40/results.csv"
        model_run = 0

    n = pypsa.Network(network)

    results = pd.read_csv(results_f).fillna("")

    df = extract_results(n, results)

    df["run"] = int(model_run)

    df.to_csv(csv, index=False)
