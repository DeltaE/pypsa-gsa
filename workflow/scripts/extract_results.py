"""Extracts results to have SA run over"""

import pandas as pd
import pypsa

import logging
logger = logging.getLogger(__name__)

def _get_p_nom_opt(n: pypsa.Network, component: str, carriers: list[str]) -> float:
    df = getattr(n,component)
    return df[df.carrier.isin(carriers)].p_nom_opt.sum()

def _get_p_total(n: pypsa.Network, component: str, var: str, carriers: list[str]) -> float:
    
    static_component = component.split("_t")[0]
    static = getattr(n,static_component)
    slicer = static[static.carrier.isin(carriers)].index
    
    year = n.investment_periods[0] # already checked that len == 1
    df = getattr(n,component)[var].loc[year]
    df = df[slicer]
    
    if var in ("p1", "p2"):
        assert all(df <= 0)
        df = df.mul(-1)
    
    sns_weights = n.snapshot_weightings.loc[year].objective
    return df.mul(sns_weights, axis=0).sum().sum()

def _get_marginal_cost(n: pypsa.Network, carriers: list[str], metric: str = "mean") -> float:
    assert metric in ("mean", "std", "min", "25%", "50%", "75%", "max")
    buses = n.buses[n.buses.carrier.isin(carriers)].index.to_list()
    return n.buses_t["marginal_price"][buses].mean(axis=1).describe().loc[metric]

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

    data = []

    for name, row in res.iterrows():

        component = row["component"]
        variable = row["variable"]
        carriers = row["carriers"]

        if variable == "p_nom_opt":
            value = _get_p_nom_opt(n, component, carriers)
        elif variable in ("p", "p0", "p1", "p2"):
            value = _get_p_total(n, component, variable, carriers)
        elif variable == "cost":
            value = _get_objective_cost(n)
        elif variable == "marginal_price":
            value = _get_marginal_cost(n, carriers, metric="mean")
        else:
            raise KeyError(f"Unrecognized argument of {variable}.")

        data.append([name, value])

    return pd.DataFrame(data, columns=["name", "value"])


if __name__ == "__main__":
    if "snakemake" in globals():
        network = snakemake.input.network
        results_f = snakemake.input.results
        model_run = snakemake.wildcards.run
        csv = snakemake.output.csv
    else:
        network = "results/Testing/gsa/modelruns/40/network.nc"
        results_f = "config/results.csv"
        csv = "results/Testing/gsa/modelruns/40/results.csv"
        model_run = 0

    n = pypsa.Network(network)

    results = pd.read_csv(results_f).fillna("")

    df = extract_results(n, results)

    df["run"] = int(model_run)

    df.to_csv(csv, index=False)
