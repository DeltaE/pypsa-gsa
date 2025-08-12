"""Extracts results to have SA run over"""

import pandas as pd
import numpy as np
import pypsa
from utils import configure_logging

import logging

logger = logging.getLogger(__name__)


def _get_p_nom_opt(n: pypsa.Network, component: str, carriers: list[str]) -> float:
    df = getattr(n, component)
    return df[df.carrier.isin(carriers)].p_nom_opt.sum()


def _get_p_nom_new(n: pypsa.Network, component: str, carriers: list[str]) -> float:
    df = getattr(n, component)
    original = df[df.carrier.isin(carriers)].p_nom.sum()
    optimial = df[df.carrier.isin(carriers)].p_nom_opt.sum()
    return optimial - original


def _get_e_nom_opt(n: pypsa.Network, component: str, carriers: list[str]) -> float:
    df = getattr(n, component)
    e_nom_opt = df[df.carrier.isin(carriers)].e_nom_opt.sum()
    if e_nom_opt == np.inf:
        assert component == "stores"
        stores = df[df.carrier.isin(carriers)].index
        df = getattr(n, "stores_t")["e"][stores]
        # emissions just want final value
        if any([x in carriers for x in ["co2", "ch4"]]):
            assert all(df >= 0)
            return round(df.max().sum() * 1e-6, 3)  # convert to MMT
        else:
            return df.sum().sum()
    else:
        return e_nom_opt


def _get_p_total(
    n: pypsa.Network, component: str, var: str, carriers: list[str]
) -> float:
    static_component = component.split("_t")[0]
    static = getattr(n, static_component)
    slicer = static[static.carrier.isin(carriers)].index

    year = n.investment_periods[0]  # already checked that len == 1
    df = getattr(n, component)[var].loc[year]
    df = df[slicer]

    if var in ("p1", "p2"):
        assert all(df <= 0)
        df = df.mul(-1)

    sns_weights = n.snapshot_weightings.loc[year].objective
    return df.mul(sns_weights, axis=0).sum().sum()


def _get_marginal_cost(
    n: pypsa.Network, carriers: list[str], metric: str = "mean"
) -> float:
    assert metric in ("mean", "std", "min", "25%", "50%", "75%", "max")
    buses = n.buses[n.buses.carrier.isin(carriers)].index.to_list()
    return n.buses_t["marginal_price"][buses].mean(axis=1).describe().loc[metric]


def _get_utilization_rate(n: pypsa.Network, component: str, carriers: list[str]) -> float:
    if component == "generators_t":
        actual = _get_generator_actual_output(n, carriers)
        maximum = _get_generator_maximum_output(n, carriers)
    elif component == "links_t":
        actual = _get_link_actual_output(n, carriers)
        maximum = _get_link_maximum_output(n, carriers)
    else:
        raise ValueError(
            f"Unrecognized component: {component}. Must be one of generators_t or links_t."
        )

    utilization_rate = actual / maximum * 100

    assert 0 <= utilization_rate <= 100, (
        f"Utilization rate is {utilization_rate} which is not between 0 and 100 for carriers {carriers}."
    )

    return round(utilization_rate, 3)


def _get_generator_actual_output(n: pypsa.Network, carriers: list[str]) -> float:
    gens = n.generators[n.generators.carrier.isin(carriers)].index
    return (
        n.generators_t["p"][gens]
        .mul(n.snapshot_weightings.objective, axis=0)
        .sum()
        .sum()
    )


def _get_generator_maximum_output(n: pypsa.Network, carriers: list[str]) -> float:
    gens = n.generators[n.generators.carrier.isin(carriers)].index
    return (
        n.generators["p_nom_opt"][gens]
        .mul(n.snapshot_weightings.objective.sum(), axis=0)
        .sum()
        .sum()
    )


def _get_link_actual_output(n: pypsa.Network, carriers: list[str]) -> float:
    links = n.links[n.links.carrier.isin(carriers)].index
    gen = (
        n.links_t.p1[links]
        .mul(-1)
        .mul(n.snapshot_weightings.objective, axis=0)
        .sum()
        .sum()
    )
    for car in carriers:  # get hp cooling generation
        if any(x in car for x in ("ashp", "gshp")):
            additional_gen = (
                n.links_t.p2[car]
                .mul(-1)
                .mul(n.snapshot_weightings.objective, axis=0)
                .sum()
                .sum()
            )
            gen += additional_gen
    return gen


def _get_link_maximum_output(n: pypsa.Network, carriers: list[str]) -> float:
    links = n.links[n.links.carrier.isin(carriers)].index

    eff = n.get_switchable_as_dense("Link", "efficiency")
    eff = eff[[x for x in eff.columns if x in links]]

    cap = n.links.loc[eff.columns].p_nom_opt
    return eff.mul(cap).mul(n.snapshot_weightings.objective, axis=0).sum().sum()


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
        elif variable == "p_nom_new":
            value = _get_p_nom_new(n, component, carriers)
        elif variable in ("p", "p0", "p1", "p2"):
            value = _get_p_total(n, component, variable, carriers)
        elif variable == "cost":
            value = _get_objective_cost(n)
        elif variable.startswith("marginal_price"):
            metric = variable.split("_")[-1]
            if metric == "price":
                metric = "mean"
            elif metric in ("25", "50", "75"):
                metric = f"{metric}%"
            if metric == "iqr":  # have to manually calculate iqr
                upper = _get_marginal_cost(n, carriers, metric="75%")
                lower = _get_marginal_cost(n, carriers, metric="25%")
                value = upper - lower
            else:
                value = _get_marginal_cost(n, carriers, metric=metric)
        elif variable == "e_nom_opt":
            value = _get_e_nom_opt(n, component, carriers)
        elif variable == "utilization":
            value = _get_utilization_rate(n, component, carriers)
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
        configure_logging(snakemake)
    else:
        network = "results/caiso/gsa/modelruns/0/network.nc"
        results_f = "results/caiso/gsa/results.csv"
        csv = "results/caiso/gsa/modelruns/0/results.csv"
        model_run = 1678

    n = pypsa.Network(network)

    results = pd.read_csv(results_f).fillna("")

    df = extract_results(n, results)

    df["run"] = int(model_run)

    df.to_csv(csv, index=False)
