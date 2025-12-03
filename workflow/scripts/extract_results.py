"""Extracts results to have SA run over"""

import pandas as pd
from math import isclose
import numpy as np
import pypsa
from utils import configure_logging

import logging

logger = logging.getLogger(__name__)


def _get_p_nom_opt(n: pypsa.Network, component: str, carriers: list[str]) -> float:
    df = getattr(n, component)
    if component == "storage_units":
        return df[df.carrier.isin(carriers)].p_nom_opt.mul(df["max_hours"]).sum()
    else:
        return df[df.carrier.isin(carriers)].p_nom_opt.sum().round(3)


def _get_p_nom_new(n: pypsa.Network, component: str, carriers: list[str]) -> float:
    df = getattr(n, component)
    if component == "storage_units":
        original = df[df.carrier.isin(carriers)].p_nom.mul(df["max_hours"]).sum()
        optimial = df[df.carrier.isin(carriers)].p_nom_opt.mul(df["max_hours"]).sum()
    else:
        original = df[df.carrier.isin(carriers)].p_nom.sum()
        optimial = df[df.carrier.isin(carriers)].p_nom_opt.sum()
    return round(optimial - original, 3)


def _get_e_nom_opt_gas_trade(n: pypsa.Network, component: str, carrier: str) -> float:
    """Handle edge case of natural gas trade not indexed by imports/exports"""
    df = getattr(n, component)
    stores = df[df.carrier.isin(["gas trade"])].copy()
    # buses are in the format 'CA AZ gas trade' and 'AZ CA gas trade'
    stores["from"] = stores.bus.str.split(" ").str[0]
    stores["to"] = stores.bus.str.split(" ").str[1]
    internal_buses = n.buses[n.buses.carrier == "gas"].country.unique().tolist()
    if carrier == "gas exports":
        return stores[stores["from"].isin(internal_buses)].e_nom_opt.sum()
    elif carrier == "gas imports":
        return stores[stores["to"].isin(internal_buses)].e_nom_opt.sum()


def _get_e_nom_opt(n: pypsa.Network, component: str, carriers: list[str]) -> float:
    df = getattr(n, component)
    # edge case of natural gas trade not indexed by imports/exports
    if any([x in carriers for x in ["gas imports", "gas exports"]]):
        if len(carriers) == 1:
            e_nom_opt = _get_e_nom_opt_gas_trade(n, component, carriers[0])
        else:
            e_nom_opt = df[df.carrier.isin(carriers)].e_nom_opt.sum()
    else:
        e_nom_opt = df[df.carrier.isin(carriers)].e_nom_opt.sum()
    if e_nom_opt == np.inf:
        assert component == "stores"
        stores = df[df.carrier.isin(carriers)].index
        df = getattr(n, "stores_t")["e"][stores]
        # emissions just want final value
        if any([x in carriers for x in ["co2", "ch4"]]):
            assert all(df >= 0)
            return round(df.max().sum() * 1e-6, 3)  # convert to MMT
        if any([x == "demand_response" for x in carriers]):
            # correct for backwards demand response having negative values
            df = df.abs()
            return float(df.sum().sum())
        else:
            return df.sum().sum()
    else:
        return e_nom_opt


def _get_e_nom_metric(
    n: pypsa.Network, component: str, carriers: list[str], metric: str
) -> float:
    assert all(
        x in ["demand_response", "gas storage", "gas pipeline"] for x in carriers
    ), f"Received carriers: {carriers}"
    df = getattr(n, component)
    stores = df[df.carrier.isin(carriers)].index
    if metric == "max":
        return n.stores_t["e"][stores].abs().sum(axis=1).max()
    elif metric == "avg":
        return float(n.stores_t["e"][stores].abs().mean().mean())
    else:
        raise ValueError(
            f"Unrecognized metric: {metric}. Must be one of ['max', 'avg']."
        )


def _get_p_total(
    n: pypsa.Network, component: str, var: str, carriers: list[str]
) -> float:
    static_component = component.split("_t")[0]
    static = getattr(n, static_component)
    slicer = static[static.carrier.isin(carriers)].index

    year = n.investment_periods[0]  # already checked that len == 1
    df = getattr(n, component)[var].loc[year]

    if df.empty:
        return 0

    df = df[slicer]

    # only get power injected into the grid
    if component == "storage_units_t":
        df = df.where(df > 0, 0)

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


def _get_utilization_rate(
    n: pypsa.Network, component: str, carriers: list[str]
) -> float:
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

    if isclose(maximum, 0, abs_tol=1e-6):
        return 0

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
    p0 = n.links_t.p0[links]
    p0 = p0.where(p0 > 0, 0)  # feasability tolerance issues on edge cases
    eff = n.get_switchable_as_dense("Link", "efficiency")
    eff = eff[[x for x in eff.columns if x in links]]
    gen = p0.mul(eff).mul(n.snapshot_weightings.objective, axis=0).sum().sum()
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
        elif variable == "e_nom_max":
            # edge case for demand response and nat gas storage
            value = _get_e_nom_metric(n, component, carriers, "max")
        elif variable == "e_nom_avg":
            # edge case for demand response and nat gas storage
            value = _get_e_nom_metric(n, component, carriers, "avg")
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
        network = "results/nd/gsa/modelruns/408/network.nc"
        results_f = "results/nd/gsa/results.csv"
        csv = "results/nd/gsa/modelruns/408/results.csv"
        model_run = 408

    n = pypsa.Network(network)

    results = pd.read_csv(results_f).fillna("")

    df = extract_results(n, results)

    df["run"] = int(model_run)

    df.to_csv(csv, index=False)
