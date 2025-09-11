"""Applies electrical imports and exports to the network."""

import numpy as np
import pypsa
import pandas as pd
from pathlib import Path


def add_import_export_carriers(n: pypsa.Network) -> None:
    """Adds import and export carriers to the network."""
    n.add("Carrier", "imports", co2_emissions=0, nice_name="Electrical Imports")
    n.add("Carrier", "exports", co2_emissions=0, nice_name="Electrical Exports")


def add_import_export_buses(n: pypsa.Network, regions_2_add: list[str]) -> None:
    """Adds import and export buses to the network."""
    n.madd(
        "Bus",
        regions_2_add,
        suffix="_imports",
        carrier="imports",
        country=regions_2_add,
    )
    n.madd(
        "Bus",
        regions_2_add,
        suffix="_exports",
        carrier="exports",
        country=regions_2_add,
    )


def add_import_export_stores(n: pypsa.Network, regions_2_add: list[str]) -> None:
    """Adds import and export stores to the network."""
    n.madd(
        "Store",
        regions_2_add,
        bus=[f"{x}_imports" for x in regions_2_add],
        suffix="_imports",
        carrier="imports",
        e_nom_extendable=True,
        marginal_cost=0,
        e_nom=0,
        e_nom_max=np.inf,
        e_min=0,
        e_min_pu=-1,
        e_max_pu=0,
    )
    n.madd(
        "Store",
        regions_2_add,
        bus=[f"{x}_exports" for x in regions_2_add],
        suffix="_exports",
        carrier="exports",
        e_nom_extendable=True,
        marginal_cost=0,
        e_nom=0,
        e_nom_max=np.inf,
        e_min=0,
        e_min_pu=0,
        e_max_pu=1,
    )


def _build_cost_timeseries(
    sns: pd.DatetimeIndex, import_costs: pd.DataFrame, r: str
) -> pd.Series:
    """Builds a cost timeseries for a given ISO."""
    timesteps = sns.get_level_values("timestep")
    years = sns.get_level_values("period").unique()
    assert len(years) == 1
    year = years[0]
    df = import_costs[import_costs.state == r].drop(columns=["state", "units"])
    df.index = pd.to_datetime(df.index).map(lambda x: x.replace(year=year))
    df = df.resample("h").ffill().reindex(timesteps).ffill()
    df.index = sns
    return df


def add_import_export_links(
    n: pypsa.Network,
    flowgates: pd.DataFrame,
    import_costs: pd.DataFrame,
    reeds_2_states: pd.DataFrame,
) -> None:
    """Adds import and export links to the network."""

    costs = {}
    sns = n.snapshots

    for _, row in flowgates.iterrows():  # super slow but works for now
        r = row.r
        rr = row.rr

        state = reeds_2_states[rr]  # get import/export costs from neighbouring state
        if state not in costs:  # extremely crude cashing :|
            costs[state] = _build_cost_timeseries(sns, import_costs, state)
        marginal_cost = costs[state]

        import_capacity = row.MW_f0
        export_capacity = row.MW_r0

        emission_bus = n.stores[
            (n.stores.carrier.str.contains("co2")) & (n.stores.bus.str.contains("pwr-"))
        ]
        assert len(emission_bus) == 1
        emission_bus = emission_bus.bus.values[0]

        n.add(
            "Link",
            f"{r}_{rr}_imports",
            bus0=f"{rr}_imports",
            bus1=r,
            bus2=emission_bus,
            carrier="imports",
            p_nom_extendable=False,
            p_min_pu=0,
            p_max_pu=1,
            marginal_cost=marginal_cost.value,
            p_nom=import_capacity,
            efficiency=1,
            efficiency2=0.384,  # T / MWh
        )

        n.add(
            "Link",
            f"{r}_{rr}_exports",
            bus0=r,
            bus1=f"{rr}_exports",
            carrier="exports",
            p_nom_extendable=False,
            p_min_pu=0,
            p_max_pu=1,
            marginal_cost=marginal_cost.value.mul(-1),  # constraint will limit exports
            p_nom=export_capacity,
        )


if __name__ == "__main__":
    if "snakemake" in globals():
        network_in = snakemake.input.network
        capacities_f = snakemake.input.capacities_f
        elec_costs_f = snakemake.input.elec_costs_f
        membership_f = snakemake.input.membership_f
        network_out = snakemake.output.network
    else:
        network_in = Path("results", "caiso", "base.nc")
        capacities_f = "capacities.csv"
        elec_costs_f = "elec_costs.csv"
        membership_f = Path("resources", "interchanges", "membership.csv")
        network_out = Path("results", "caiso", "imports_exports.nc")

    flowgates = pd.read_csv(capacities_f)
    regions_2_add = flowgates.rr.unique().tolist()

    import_costs = pd.read_csv(elec_costs_f, index_col=0)
    membership = pd.read_csv(membership_f, index_col=0)
    reeds_2_states = membership.st.to_dict()

    n = pypsa.Network(network_in)
    add_import_export_carriers(n)
    add_import_export_buses(n, regions_2_add)
    add_import_export_stores(n, regions_2_add)
    add_import_export_links(n, flowgates, import_costs, reeds_2_states)

    n.export_to_netcdf(network_out)
