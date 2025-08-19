"""
Modified solve network file from PyPSA-USA

https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/solve_network.py
"""

import numpy as np
import pandas as pd
import pypsa

from typing import Optional
import yaml
from utils import (
    get_existing_lv,
    get_network_iso,
    get_region_buses,
    get_rps_demand_supplyside,
    get_rps_eligible,
    get_rps_generation,
    concat_rps_standards,
    format_raw_ng_trade_data,
    get_ng_trade_links,
    get_urban_rural_fraction,
    configure_logging,
)

import logging

logger = logging.getLogger(__name__)

###
# Helpers
###


def filter_components(
    n: pypsa.Network,
    component_type: str,
    planning_horizon: str | int,
    carrier_list: list[str],
    region_buses: pd.Index | str,
    extendable: bool,
):
    """
    Filter components based on common criteria.

    Parameters
    ----------
    - n: pypsa.Network
        The PyPSA network object.
    - component_type: str
        The type of component (e.g., "Generator", "StorageUnit").
    - planning_horizon: str or int
        The planning horizon to filter active assets.
    - carrier_list: list
        List of carriers to filter.
    - region_buses: pd.Index
        Index of region buses to filter.
    - extendable: bool, optional
        If specified, filters by extendable or non-extendable assets.

    Returns
    -------
    - pd.DataFrame
        Filtered assets.
    """
    component = n.df(component_type)
    if planning_horizon != "all":
        ph = int(planning_horizon)
        iv = n.investment_periods
        active_components = n.get_active_assets(component.index.name, iv[iv >= ph][0])
    else:
        active_components = component.index

    if isinstance(region_buses, str):
        if region_buses == "all":
            region_buses = n.buses.index
    assert isinstance(region_buses, pd.Index)

    # Links will throw the following attribute error, as we must specify bus0
    # AttributeError: 'DataFrame' object has no attribute 'bus'. Did you mean: 'bus0'?
    bus_name = "bus0" if component_type.lower() == "link" else "bus"

    filtered = component.loc[
        active_components
        & component.carrier.isin(carrier_list)
        & component[bus_name].isin(region_buses)
        & (component.p_nom_extendable == extendable)
    ]

    return filtered


###
# Custom constraints
###


def add_land_use_constraint_perfect(n):
    """
    Add global constraints for tech capacity limit.
    """

    def check_p_min_p_max(p_nom_max):
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            raise ValueError(
                f"p_min_pu values at node larger than technical potential {check[check].index}"
            )

    grouper = [n.generators.carrier, n.generators.bus]
    ext_i = n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing")

    # get technical limit per node
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max

    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]

    # carrier
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0

    # check minimum capacities
    check_p_min_p_max(p_nom_max)

    df = p_nom_max.reset_index()
    df["name"] = df.apply(lambda row: f"nom_max_{row['carrier']}", axis=1)

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values
    return n


def add_no_coal_oil_investment_constraint(n):
    """
    Add constraint to prevent investment in coal and oil.
    """
    cars = [
        x
        for x in n.carriers.index
        if any(fuel in x for fuel in ["coal", "oil", "waste"])
    ]
    n.links.loc[n.links.carrier.isin(cars), "p_nom_extendable"] = False
    return n


def no_economic_retirement_constraint(n):
    """
    Turns off economic retirement for power generator assets.

    This can be configured in the main pypsa_usa workflow. Its done here though as it
    allows faster testing of economic retirement.
    """

    # very hacky, but I know this wont change for my project
    pwr_cars = [
        "nuclear",
        "oil",
        "OCGT",
        "CCGT",
        "CCGT-95CCS",
        "coal",
        "geothermal",
        "biomass",
        "waste",
        "onwind",
        "offwind_floating",
        "solar",
        "hydro",
        "battery",
    ]

    links = n.links[
        (n.links.index.str.endswith(" existing")) & (n.links.carrier.isin(pwr_cars))
    ]
    n.links.loc[links.index, "p_nom_extendable"] = False
    n.links.loc[links.index, "capital_cost"] = (
        0  # not actually needed, just for sanity :)
    )

    gens = n.generators[
        (n.generators.index.str.endswith(" existing"))
        & (n.generators.carrier.isin(pwr_cars))
    ]
    n.generators.loc[gens.index, "p_nom_extendable"] = False
    n.generators.loc[gens.index, "capital_cost"] = (
        0  # not actually needed, just for sanity :)
    )

    storageunits = n.storage_units[
        (n.storage_units.index.str.endswith(" existing"))
        & (n.storage_units.carrier.isin(pwr_cars))
    ]
    n.storage_units.loc[storageunits.index, "p_nom_extendable"] = False
    n.storage_units.loc[storageunits.index, "capital_cost"] = (
        0  # not actually needed, just for sanity :)
    )
    return n


def add_technology_capacity_target_constraints(
    n: pypsa.Network, data: pd.DataFrame, sample: pd.DataFrame
):
    """
    Add Technology Capacity Target (TCT) constraint to the network.

    Add minimum or maximum levels of generator nominal capacity per carrier for individual regions.
    Each constraint can be designated for a specified planning horizon in multi-period models.
    Opts and path for technology_capacity_targets.csv must be defined in config.yaml.
    Default file is available at config/policy_constraints/technology_capacity_targets.csv.
    """

    tct_data = data.copy()

    # apply sample
    tct_data = tct_data.set_index("name")
    for name, sample_value in zip(sample.name, sample.value):
        tct_data.loc[name, "max"] = sample_value  # sample value already scaled
    tct_data = tct_data.reset_index()

    if tct_data.empty:
        return

    for _, target in tct_data.iterrows():
        planning_horizon = target.planning_horizon
        region_list = [region_.strip() for region_ in target.region.split(",")]
        carrier_list = [carrier_.strip() for carrier_ in target.carrier.split(",")]
        region_buses = get_region_buses(n, region_list)

        lhs_gens_ext = filter_components(
            n=n,
            component_type="Generator",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=True,
        )
        lhs_gens_existing = filter_components(
            n=n,
            component_type="Generator",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=False,
        )

        lhs_storage_ext = filter_components(
            n=n,
            component_type="StorageUnit",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=True,
        )
        lhs_storage_existing = filter_components(
            n=n,
            component_type="StorageUnit",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=False,
        )

        lhs_link_ext = filter_components(
            n=n,
            component_type="Link",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=True,
        )
        lhs_link_existing = filter_components(
            n=n,
            component_type="Link",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=False,
        )

        if region_buses.empty or (
            lhs_gens_ext.empty and lhs_storage_ext.empty and lhs_link_ext.empty
        ):
            continue

        if not lhs_gens_ext.empty:
            grouper_g = pd.concat(
                [lhs_gens_ext.bus.map(n.buses.country), lhs_gens_ext.carrier],
                axis=1,
            ).rename_axis(
                "Generator-ext",
            )
            lhs_g = (
                n.model["Generator-p_nom"]
                .loc[lhs_gens_ext.index]
                .groupby(grouper_g)
                .sum()
                .rename(bus="country")
            )
        else:
            lhs_g = None

        if not lhs_storage_ext.empty:
            grouper_s = pd.concat(
                [lhs_storage_ext.bus.map(n.buses.country), lhs_storage_ext.carrier],
                axis=1,
            ).rename_axis(
                "StorageUnit-ext",
            )
            lhs_s = (
                n.model["StorageUnit-p_nom"]
                .loc[lhs_storage_ext.index]
                .groupby(grouper_s)
                .sum()
            )
        else:
            lhs_s = None

        if not lhs_link_ext.empty:
            grouper_l = pd.concat(
                [lhs_link_ext.bus1.map(n.buses.country), lhs_link_ext.carrier],
                axis=1,
            ).rename_axis(
                "Link-ext",
            )
            lhs_l = (
                n.model["Link-p_nom"].loc[lhs_link_ext.index].groupby(grouper_l).sum()
            )
        else:
            lhs_l = None

        if lhs_g is None and lhs_s is None and lhs_l is None:
            continue
        else:
            gen = lhs_g.sum() if lhs_g else 0
            lnk = lhs_l.sum() if lhs_l else 0
            sto = lhs_s.sum() if lhs_s else 0

        lhs = gen + lnk + sto

        lhs_existing = (
            lhs_gens_existing.p_nom.sum()
            + lhs_storage_existing.p_nom.sum()
            + lhs_link_existing.p_nom.sum()
        )

        if target["max"] == "existing":
            target["max"] = round(lhs_existing, 5) + 0.01
        else:
            target["max"] = float(target["max"])

        if target["min"] == "existing":
            target["min"] = round(lhs_existing, 5) - 0.01
        else:
            target["min"] = float(target["min"])

        if not np.isnan(target["min"]):
            rhs = target["min"] - round(lhs_existing, 5)

            n.model.add_constraints(
                lhs >= rhs,
                name=f"GlobalConstraint-{target['name']}_{target['planning_horizon']}_min",
            )

            logger.info(
                "Adding TCT Constraint:\n"
                f"Name: {target.name}\n"
                f"Planning Horizon: {target.planning_horizon}\n"
                f"Region: {target.region}\n"
                f"Carrier: {target.carrier}\n"
                f"Min Value: {target['min']}\n"
                f"Min Value Adj: {rhs}",
            )

        if not np.isnan(target["max"]):
            assert target["max"] >= lhs_existing, (
                f"TCT constraint of {target['max']} MW for {target['carrier']} must be at least {lhs_existing}"
            )

            rhs = target["max"] - round(lhs_existing, 5)

            n.model.add_constraints(
                lhs <= rhs,
                name=f"GlobalConstraint-{target['name']}_{target['planning_horizon']}_max",
            )

            logger.info(
                "Adding TCT Constraint:\n"
                f"Name: {target.name}\n"
                f"Planning Horizon: {target.planning_horizon}\n"
                f"Region: {target.region}\n"
                f"Carrier: {target.carrier}\n"
                f"Max Value: {target['max']}\n"
                f"Max Value Adj: {rhs}",
            )


def add_RPS_constraints(
    n: pypsa.Network, policy_name: str, rps: pd.DataFrame, sample: float
):
    """
    Add Renewable Portfolio Standards constraint to the network.

    This is applied at a **supply level**, not a demand level. Else we need to account for
    imports/exports and sector links (ie. heatpumps, ect).
    """

    if rps.empty:
        return

    portfolio_standards = concat_rps_standards(n, rps)

    mapper = n.buses.groupby("reeds_state")["rec_trading_zone"].first().to_dict()
    portfolio_standards["rec_trading_zone"] = portfolio_standards.region.map(
        mapper
    ).fillna(portfolio_standards.region)

    for rec_trading_zone in portfolio_standards.rec_trading_zone.unique():
        portfolio_standards_zone = portfolio_standards[
            portfolio_standards.rec_trading_zone == rec_trading_zone
        ]

        demands = []  # linear expressions for each demand
        generation = []  # linear expressions for each generation

        for _, constraint_row in portfolio_standards_zone.iterrows():
            region_buses, region_gens = get_rps_eligible(
                n, constraint_row.region, constraint_row.carrier
            )

            if region_buses.empty:
                continue

            if not region_gens.empty:
                region_demand = get_rps_demand_supplyside(
                    n, constraint_row.planning_horizon, region_buses, region_gens
                )
                # region_demand = get_rps_demand_demandside(
                #     n, constraint_row.planning_horizon, region_buses
                # )

                # pct is really a decimal value, not a percentage
                demands.append(constraint_row.pct * region_demand * sample)

                region_gen = get_rps_generation(
                    n, constraint_row.planning_horizon, region_gens
                )
                generation.append(region_gen)

        demand = sum(demands)
        generation = sum(generation)

        lhs = generation - demand
        rhs = 0

        # Add constraint
        n.model.add_constraints(
            lhs >= rhs,
            name=f"GlobalConstraint-{constraint_row.name}_{constraint_row.planning_horizon}_{policy_name}_limit",
        )
        logger.info(
            f"Added {rec_trading_zone} {policy_name} for {constraint_row.planning_horizon}.",
        )


def add_sector_co2_constraints(
    n: pypsa.Network, sample: float, include_ch4: bool = True
):
    """Adds sector co2 constraints."""

    def apply_national_limit(
        n: pypsa.Network,
        year: int,
        value: float,
        include_ch4: bool,
        sector: str | None = None,
    ):
        """For every snapshot, sum of co2 and ch4 must be less than limit."""
        if sector:
            if include_ch4:
                stores = n.stores[
                    (
                        (n.stores.index.str.endswith(f"{sector}-co2"))
                        | (n.stores.index.str.endswith(f"{sector}-ch4"))
                    )
                ].index
            else:
                stores = n.stores[n.stores.index.str.endswith(f"{sector}-co2")].index
            name = f"co2_limit-{year}-{sector}"
            log_statement = f"Adding national {sector} co2 Limit in {year} of"
        else:
            if include_ch4:
                stores = n.stores[
                    (
                        (n.stores.index.str.endswith("-co2"))
                        | (n.stores.index.str.endswith("-ch4"))
                    )
                ].index
            else:
                stores = n.stores[n.stores.index.str.endswith("-co2")].index
            name = f"co2_limit-{year}"
            log_statement = f"Adding national co2 Limit in {year} of"

        lhs = n.model["Store-e"].loc[:, stores].sum(dim="Store")
        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=name)

        logger.info(f"{log_statement} {rhs * 1e-6} MMT CO2")

    # limit is applied at a global level

    mmt_limit = sample
    year = n.investment_periods[0]

    df = pd.DataFrame(
        [[year, "all", "all", mmt_limit]],
        columns=["year", "state", "sector", "co2_limit_mmt"],
    )

    if df.empty:
        logger.debug("No co2 policies applied")
        return

    sectors = df.sector.unique()

    for sector in sectors:
        df_sector = df[df.sector == sector]
        states = df_sector.state.unique()

        for state in states:
            df_state = df_sector[df_sector.state == state]
            years = [x for x in df_state.year.unique() if x in n.investment_periods]

            if not years:
                logger.debug(
                    f"No co2 policies applied for {sector} due to no defined years",
                )
                continue

            for year in years:
                df_limit = df_state[df_state.year == year].reset_index(drop=True)
                assert df_limit.shape[0] == 1

                # results calcualted in T CO2, policy given in MMT CO2
                value = df_limit.loc[0, "co2_limit_mmt"] * 1e6

                if state.lower() == "all":
                    if sector == "all":
                        apply_national_limit(n, year, value, include_ch4)
                    else:
                        apply_national_limit(n, year, value, include_ch4, sector)
                else:
                    raise ValueError(state.lower())


def add_ng_import_export_limits(
    n: pypsa.Network, ng_trade: dict[str, pd.DataFrame | float]
):
    def add_import_limits(n, data, constraint, multiplier=None):
        """Sets gas import limit over each year."""
        assert constraint in ("max", "min")

        if not multiplier:
            multiplier = 1

        weights = n.snapshot_weightings.objective

        links = get_ng_trade_links(n, "imports")

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = data.at[link, "rhs"] * multiplier
                except KeyError:
                    # logger.debug(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                if constraint == "min":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"ng_limit_import_min-{year}-{link}",
                    )
                else:
                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"ng_limit_import_max-{year}-{link}",
                    )

    def add_export_limits(n, data, constraint, multiplier=None):
        """Sets maximum export limit over the year."""
        assert constraint in ("max", "min")

        if not multiplier:
            multiplier = 1

        weights = n.snapshot_weightings.objective

        links = get_ng_trade_links(n, "exports")

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = data.at[link, "rhs"] * multiplier
                except KeyError:
                    # logger.debug(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                if constraint == "min":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"ng_limit_export_min-{year}-{link}",
                    )
                else:
                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"ng_limit_export_max-{year}-{link}",
                    )

    # get limits

    import_min = round(ng_trade.get("min_import", 1), 3)
    import_max = round(ng_trade.get("max_import", 1), 3)
    export_min = round(ng_trade.get("min_export", 1), 3)
    export_max = round(ng_trade.get("max_export", 1), 3)
    # to avoid numerical issues, ensure there is a gap between min/max constraints
    if abs(import_max - import_min) < 0.0001:
        import_min -= 0.01
        import_max += 0.01
        if import_min < 0:
            import_min = 0

    if abs(export_max - export_min) < 0.0001:
        export_min -= 0.01
        export_max += 0.01
        if export_min < 0:
            export_min = 0

    # import and export dataframes contain the same information, just in different formats
    # ie. imports from one S1 -> S2 are the same as exports from S2 -> S1
    # we use the exports direction to set limits

    # add domestic limits

    trade = ng_trade["domestic"].copy()
    trade = format_raw_ng_trade_data(trade, " trade")

    add_import_limits(n, trade, "min", import_min)
    add_export_limits(n, trade, "min", export_min)

    if not import_max == "inf":
        add_import_limits(n, trade, "max", import_max)
    if not export_max == "inf":
        add_export_limits(n, trade, "max", export_max)

    # add international limits

    trade = ng_trade["international"].copy()
    trade = format_raw_ng_trade_data(trade, " trade")

    add_import_limits(n, trade, "min", import_min)
    add_export_limits(n, trade, "min", export_min)

    if not import_max == "inf":
        add_import_limits(n, trade, "max", import_max)
    if not export_max == "inf":
        add_export_limits(n, trade, "max", export_max)


def add_elec_trade_constraints(n: pypsa.Network, elec_trade: pd.DataFrame):
    def _get_elec_import_links(n: pypsa.Network, iso: str) -> list[str]:
        """Get all links for elec trade."""
        return n.links[
            (n.links.carrier == "imports") & (n.links.bus0.str.startswith(iso))
        ].index

    def _get_elec_export_links(n: pypsa.Network, iso: str) -> list[str]:
        """Get all links for elec trade."""
        return n.links[
            (n.links.carrier == "exports") & (n.links.bus1.str.startswith(iso))
        ].index

    from_iso = get_network_iso(n)
    if len(from_iso) < 1:
        raise ValueError("No full ISOs found for network")
    elif len(from_iso) > 1:
        raise ValueError("Multiple ISOs found for network")
    from_iso = from_iso[0]

    # get unique to isos for constraint setup
    flows = elec_trade[
        ((elec_trade["to"] == from_iso) | (elec_trade["from"] == from_iso))
        & (elec_trade["interchange_reported_mwh"] > 0)
    ]
    to_isos = list(set(flows["to"].unique().tolist() + flows["from"].unique().tolist()))
    to_isos.remove(from_iso)  # list of unique connecting isos in the network

    weights = n.snapshot_weightings.objective
    period = n.snapshots.get_level_values("period").unique().tolist()
    assert len(period) == 1, "Only one period supported for elec trade constraints"

    for to_iso in to_isos:
        import_links = _get_elec_import_links(n, to_iso)
        export_links = _get_elec_export_links(n, to_iso)

        if import_links.empty and export_links.empty:
            raise ValueError(f"No links found for {to_iso}")

        timesteps = n.snapshots.get_level_values("timestep")

        for month in flows.month.unique():
            timesteps_in_month = timesteps[timesteps.month == month].strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # filter flows to only include the current month
            rhs_df = flows[
                ((flows["from"] == to_iso) | (flows["to"] == to_iso))
                & (flows.month == month)
            ]
            assert len(rhs_df) == 1, f"Multiple flows found for {to_iso} in {month}"

            rhs = rhs_df.interchange_reported_mwh.values[0]

            imports = (
                n.model["Link-p"]
                .mul(weights)
                .sel(period=period, Link=import_links)
                .sel(
                    timestep=timesteps_in_month
                )  # Seperate cause slicing on multi-index is not supported
                .sum()
            )
            exports = (
                n.model["Link-p"]
                .mul(weights)
                .sel(period=period, Link=export_links)
                .sel(
                    timestep=timesteps_in_month
                )  # Seperate cause slicing on multi-index is not supported
                .sum()
            )

            if rhs_df["from"].values[0] == to_iso:
                # net trade from outside model scope to within model scope
                # the model can import up to the limit, but not more
                lhs = imports - exports

                n.model.add_constraints(
                    lhs <= rhs,
                    name=f"elec_trade_imports-{to_iso}-month_{month}",
                )

            elif rhs_df["to"].values[0] == to_iso:
                # net trade from inside model scope to neighboring iso
                # the model must export at least this amount
                lhs = exports - imports

                # upper and lower as exports have a negative cost
                lhs_lower = lhs.mul(0.99)
                lhs_upper = lhs.mul(1.01)

                n.model.add_constraints(
                    lhs_lower >= rhs,
                    name=f"elec_trade_exports_lower-{to_iso}-month_{month}",
                )

                n.model.add_constraints(
                    lhs_upper <= rhs,
                    name=f"elec_trade_exports_upper-{to_iso}-month_{month}",
                )

            else:
                raise ValueError(f"Invalid flow direction for {to_iso} in {month}")


def add_lolp_constraint(n):
    """Adds a loss of load probability constraint.

    This constraint restricts the amount of load shedding that can occur.
    """

    generators = n.generators[n.generators.carrier == "load"]

    if generators.empty:
        logger.warning(
            "No load shedding generators found for loss of load probability constraint"
        )
        return

    assert len(generators.bus.unique()) == len(generators), (
        "Multiple generators at the same bus"
    )

    loads = n.loads[
        n.loads.carrier.str.endswith("elec")
    ]  # llop only to electrical loads

    for generator in generators.index:
        bus = n.generators.at[generator, "bus"]
        loads_at_bus = loads[loads.country == bus].index
        load = n.loads_t["p_set"].loc[:, loads_at_bus].sum().sum()

        # LLOP of 0.1 day per year = 0.000274
        # https://en.wikipedia.org/wiki/Loss_of_load
        rhs = round(load * 0.000274, 6)

        lhs = n.model["Generator-p"].loc[:, generator].sum()

        n.model.add_constraints(
            lhs <= rhs,
            name=f"llop-{bus}",
        )


def add_transmission_limit(n, factor):
    """Set volume transmission limits expansion."""

    if factor <= 1.0:
        return

    # remove the old Global Constraint
    n.remove("GlobalConstraint", "lv_limit")

    logger.info(f"Setting volume transmission limit of {factor * 100}%")

    # ensures AC links are set to extendable.
    ac_links_extend = n.links[
        (n.links.carrier == "AC") & (n.links.index.str.endswith("exp"))
    ].index
    n.links.loc[ac_links_extend, "p_nom_extendable"] = True

    ref = get_existing_lv(n)
    rhs = float(factor) * ref

    con_type = "volume_expansion"

    n.add(
        "GlobalConstraint",
        "lv_limit",
        type=f"transmission_{con_type}_limit",
        sense="<=",
        constant=rhs,
        carrier_attribute="AC, DC",
    )


def add_cooling_heat_pump_constraints(n):
    """
    Adds constraints to the cooling heat pumps.

    These constraints allow HPs to be used to meet both heating and cooling
    demand within a single timeslice while respecting capacity limits.
    Since we are aggregating (and not modelling individual units)
    this should be fine.

    Two seperate constraints are added:
    - Constrains the cooling HP capacity to equal the heating HP capacity. Since the
    cooling hps do not have a capital cost, this will not effect objective cost
    - Constrains the total generation of Heating and Cooling HPs at each time slice
    to be less than or equal to the max generation of the heating HP. Note, that both
    the cooling and heating HPs have the same COP
    """

    def add_hp_capacity_constraint(n, hp_type):
        assert hp_type in ("ashp", "gshp")

        heating_hps = n.links[n.links.index.str.endswith(hp_type)].index
        if heating_hps.empty:
            return
        cooling_hps = n.links[n.links.index.str.endswith(f"{hp_type}-cool")].index

        assert len(heating_hps) == len(cooling_hps)

        lhs = (
            n.model["Link-p_nom"].loc[heating_hps]
            - n.model["Link-p_nom"].loc[cooling_hps]
        )
        rhs = 0

        n.model.add_constraints(lhs == rhs, name=f"Link-{hp_type}_cooling_capacity")

    def add_hp_generation_constraint(n, hp_type):
        heating_hps = n.links[n.links.index.str.endswith(hp_type)].index
        if heating_hps.empty:
            return
        cooling_hps = n.links[n.links.index.str.endswith(f"{hp_type}-cooling")].index

        heating_hp_p = n.model["Link-p"].loc[:, heating_hps]
        cooling_hp_p = n.model["Link-p"].loc[:, cooling_hps]

        heating_hps_cop = n.links_t["efficiency"][heating_hps]
        cooling_hps_cop = n.links_t["efficiency"][cooling_hps]

        heating_hps_gen = heating_hp_p.mul(heating_hps_cop)
        cooling_hps_gen = cooling_hp_p.mul(cooling_hps_cop)

        lhs = heating_hps_gen + cooling_hps_gen

        heating_hp_p_nom = n.model["Link-p_nom"].loc[heating_hps]
        max_gen = heating_hp_p_nom.mul(heating_hps_cop)

        rhs = max_gen

        n.model.add_constraints(lhs <= rhs, name=f"Link-{hp_type}_cooling_generation")

    for hp_type in ("ashp", "gshp"):
        add_hp_capacity_constraint(n, hp_type)
        add_hp_generation_constraint(n, hp_type)


def add_gshp_capacity_constraint(
    n: pypsa.Network, pop_layout: pd.DataFrame, sample: float
):
    """
    Constrains gshp capacity based on population and ashp installations.

    This constraint should be added if rural/urban sectors are combined into
    a single total area. In this case, we need to constrain how much gshp capacity
    can be added to the system.

    For example:
    - If ratio is 0.75 urban and 0.25 rural
    - We want to enforce that at max, only 0.33 unit of GSHP can be installed for every unit of ASHP
    - The constraint is: [ASHP - (urban / rural) * GSHP >= 0]
    - ie. for every unit of GSHP, we need to install 3 units of ASHP
    """

    df = pop_layout.copy()

    ashp = n.links[n.links.index.str.endswith("ashp")].copy()
    gshp = n.links[n.links.index.str.endswith("gshp")].copy()
    if gshp.empty:
        return

    assert len(ashp) == len(gshp)

    fraction = get_urban_rural_fraction(df)

    gshp["urban_rural_fraction"] = gshp.bus0.map(fraction)

    ashp_capacity = n.model["Link-p_nom"].loc[ashp.index]
    gshp_capacity = n.model["Link-p_nom"].loc[gshp.index]
    gshp_multiplier = gshp["urban_rural_fraction"].mul(sample)

    lhs = ashp_capacity - gshp_capacity.mul(gshp_multiplier.values)
    rhs = 0

    n.model.add_constraints(lhs >= rhs, name="Link-gshp_capacity_ratio")


def add_ev_generation_constraint(n, policy: pd.DataFrame, sample: float):
    mode_mapper = {
        "light_duty": "lgt",
        "med_duty": "med",
        "heavy_duty": "hvy",
        "bus": "bus",
    }
    carrier_mapper = {
        "light_duty": "trn-elec-veh-lgt",
        "med_duty": "trn-elec-veh-med",
        "heavy_duty": "trn-elec-veh-hvy",
        "bus": "trn-elec-veh-bus",
    }

    for mode in policy.columns:
        sample_mode = sample[sample.carrier == carrier_mapper[mode]]

        if len(sample_mode) < 1:  # where no ev policy is uncertain
            sample_value = 1
        else:
            assert len(sample_mode) == 1
            sample_value = sample_mode.value.values[0]

        evs = n.links[n.links.carrier == f"trn-elec-veh-{mode_mapper[mode]}"].index
        dem_names = n.loads[n.loads.carrier == f"trn-veh-{mode_mapper[mode]}"].index
        dem = n.loads_t["p_set"][dem_names]

        for investment_period in n.investment_periods:
            ratio = policy.at[investment_period, mode] / 100  # input is percentage
            eff = n.links.loc[evs].efficiency.mean()
            lhs = n.model["Link-p"].loc[investment_period].sel(Link=evs).sum()
            rhs_ref = dem.loc[investment_period].sum().sum() * ratio / eff
            rhs = rhs_ref + rhs_ref * sample_value

            n.model.add_constraints(
                lhs <= rhs, name=f"Link-ev_gen_{mode}_{investment_period}"
            )


def extra_functionality(n, sns):
    """
    Collects supplementary constraints which will be passed to `pypsa.optimization.optimize`
    """

    opts = n.extra_fn

    if "rps" in opts:
        add_RPS_constraints(n, "rps", opts["rps"]["data"], opts["rps"]["sample"])
    if "ces" in opts:
        add_RPS_constraints(n, "ces", opts["ces"]["data"], opts["ces"]["sample"])
    if "tct" in opts:
        add_technology_capacity_target_constraints(
            n, opts["tct"]["data"], opts["tct"]["sample"]
        )
    if "ev_gen" in opts:
        add_ev_generation_constraint(
            n, opts["ev_gen"]["data"], opts["ev_gen"]["sample"]
        )
    if "co2L" in opts:
        add_sector_co2_constraints(n, opts["co2L"]["sample"])
    if "gshp" in opts:
        add_gshp_capacity_constraint(n, opts["gshp"]["data"], opts["gshp"]["sample"])
    if "ng_trade" in opts:
        add_ng_import_export_limits(n, opts["ng_trade"])
    else:
        raise ValueError("No ng_limits provided")
    if "lv" in opts:
        add_transmission_limit(n, opts["lv"]["sample"])
    if "hp_cooling" in opts:
        add_cooling_heat_pump_constraints(n)
    if "elec_trade" in opts:
        add_elec_trade_constraints(n, opts["elec_trade"]["flows"])
    if "lolp" in opts:
        add_lolp_constraint(n)


###
# Prepare Network
###


def prepare_network(
    n,
    clip_p_max_pu: Optional[bool | float] = None,
    noisy_costs: Optional[bool] = None,
    foresight: Optional[str] = None,
    no_coal_oil_investment: Optional[bool] = None,
    no_economic_retirement: Optional[bool] = None,
    **kwargs,
):
    if clip_p_max_pu:
        if isinstance(clip_p_max_pu, float):
            _clip_p_max(n, clip_p_max_pu)
        else:
            _clip_p_max(n)

    if noisy_costs:
        _apply_noisy_costs(n)

    if foresight == "perfect":
        n = add_land_use_constraint_perfect(n)

    if no_coal_oil_investment:
        n = add_no_coal_oil_investment_constraint(n)

    if no_economic_retirement:
        n = no_economic_retirement_constraint(n)

    return n


def _clip_p_max(n: pypsa.Network, value: Optional[float] = None) -> None:
    if not value:
        value = 1.0e-2

    for df in (
        n.generators_t.p_max_pu,
        n.generators_t.p_min_pu,
        n.storage_units_t.inflow,
    ):
        df.where(df > value, other=0.0, inplace=True)


def _apply_noisy_costs(n: pypsa.Network) -> None:
    """Adds noise to costs"""

    for c in n.iterate_components():
        if "marginal_cost" in c.df:
            c.df["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(c.df)) - 0.5)

    for c in n.iterate_components(["Line", "Link"]):
        c.df["capital_cost"] += (
            1e-1 + 2e-2 * (np.random.random(len(c.df)) - 0.5)
        ) * c.df["length"]


###
# Sovle Network
###


def solve_network(
    n: pypsa.Network,
    solver_name: str,
    solver_options: dict[str, str],
    solving_options: dict[str, str],
    log: Optional[str] = None,
    extra_fn: Optional[dict[str, pd.DataFrame]] = None,
    **kwargs,
):
    options = {}
    options["solver_name"] = solver_name
    options["solver_options"] = solver_options
    options["transmission_losses"] = solving_options.get("transmission_losses", False)
    options["linearized_unit_commitment"] = solving_options.get(
        "linearized_unit_commitment",
        False,
    )
    options["assign_all_duals"] = solving_options.get("assign_all_duals", False)
    options["multi_investment_periods"] = True  # needed for correct lifetimes

    if log:
        options["log_fn"] = log

    # add to network for extra_functionality
    if extra_functionality:
        options["extra_functionality"] = extra_functionality
        n.extra_fn = extra_fn

    skip_iterations = solving_options.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    if skip_iterations:
        status, condition = n.optimize(**options)
    else:
        options["track_iterations"] = (solving_options.get("track_iterations", False),)
        options["min_iterations"] = (solving_options.get("min_iterations", 4),)
        options["max_iterations"] = (solving_options.get("max_iterations", 6),)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **options,
        )

    if status != "ok":
        logger.info(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    if "infeasible" in condition:
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" in globals():
        in_network = snakemake.input.network
        solver_name = snakemake.params.solver
        solver_opts = snakemake.params.solver_opts
        solving_opts = snakemake.params.solving_opts
        model_opts = snakemake.params.model_opts
        solving_log = snakemake.log.solver
        out_network = snakemake.output.network
        pop_f = snakemake.input.pop_layout_f
        ng_dommestic_f = snakemake.input.ng_domestic_f
        ng_international_f = snakemake.input.ng_international_f
        rps_f = snakemake.input.rps_f
        ces_f = snakemake.input.ces_f
        tct_f = snakemake.input.tct_f
        ev_policy_f = snakemake.input.ev_policy_f
        import_export_flows_f = snakemake.input.import_export_flows_f
        constraints_meta = snakemake.input.constraints
        configure_logging(snakemake)
    else:
        in_network = "results/caiso/gsa/modelruns/0/n.nc"
        solver_name = "gurobi"
        solving_opts_config = "config/solving.yaml"
        model_opts = {
            "economic_retirement": False,
            "coal_oil_investment": False,
        }
        solving_log = ""
        out_network = ""
        pop_f = "results/caiso/constraints/pop_layout.csv"
        ng_dommestic_f = "results/caiso/constraints/ng_domestic.csv"
        ng_international_f = "results/caiso/constraints/ng_international.csv"
        rps_f = "results/caiso/constraints/rps.csv"
        ces_f = "results/caiso/constraints/ces.csv"
        tct_f = "results/caiso/constraints/tct.csv"
        ev_policy_f = "results/caiso/constraints/ev_policy.csv"
        import_export_flows_f = "results/caiso/constraints/import_export_flows.csv"
        constraints_meta = "results/caiso/gsa/modelruns/0/constraints.csv"

        with open(solving_opts_config, "r") as f:
            solving_opts_all = yaml.safe_load(f)

        solving_opts = solving_opts_all["solving"]["options"]
        solver_opts = solving_opts_all["solving"]["solver_options"]["gurobi-default"]

    n = pypsa.Network(in_network)

    # for land use constraint
    solving_opts["foresight"] = "perfect"

    # different from pypsa-usa
    solving_opts["no_economic_retirement"] = (
        False if model_opts["economic_retirement"] else True
    )
    solving_opts["no_coal_oil_investment"] = (
        False if model_opts["coal_oil_investment"] else True
    )

    np.random.seed(solving_opts.get("seed", 123))

    n = prepare_network(n, **solving_opts)

    # holds sampled data that needs to be applied to RHS of constraints
    constraints = pd.read_csv(constraints_meta)

    extra_fn = {}

    ###
    # import/export constraints
    ###
    extra_fn["elec_trade"] = {}
    trade = pd.read_csv(import_export_flows_f)
    multipliers = constraints[constraints.attribute == "elec_trade"].round(5)
    if len(multipliers) == 1:
        trade["interchange_reported_mwh"] *= multipliers.value.values[0].round(5)
        extra_fn["elec_trade"]["flows"] = trade
    elif len(multipliers) > 1:
        raise ValueError("Too many samples for elec_trade")
    else:
        logger.debug("No elec trade multipler provided")
        extra_fn["elec_trade"]["flows"] = trade

    ###
    # natural gas constraints
    ###
    extra_fn["ng_trade"] = {}
    extra_fn["ng_trade"]["domestic"] = pd.read_csv(ng_dommestic_f, index_col=0)
    extra_fn["ng_trade"]["international"] = pd.read_csv(ng_international_f, index_col=0)
    imports = constraints[constraints.attribute == "nat_gas_import"].round(5)
    exports = constraints[constraints.attribute == "nat_gas_export"].round(5)

    # to avoid infeasibilities, we always set one limit to 1

    if len(imports) == 1:
        value = imports.value.values[0]
        if value < 1:
            extra_fn["ng_trade"]["min_import"] = value
            extra_fn["ng_trade"]["max_import"] = 1
        elif value > 1:
            extra_fn["ng_trade"]["min_import"] = 1
            extra_fn["ng_trade"]["max_import"] = value
        else:
            extra_fn["ng_trade"]["min_import"] = 0.99
            extra_fn["ng_trade"]["max_import"] = 1.01
        # extra_fn["ng_trade"]["min_import"] = 0
        # extra_fn["ng_trade"]["max_import"] = value
    elif len(imports) > 1:
        raise ValueError("Too many samples for ng_gas_import")
    else:
        extra_fn["ng_trade"]["min_import"] = 0.99
        extra_fn["ng_trade"]["max_import"] = 1.01

    if len(exports) == 1:
        value = exports.value.values[0]
        if value < 1:
            extra_fn["ng_trade"]["min_export"] = value
            extra_fn["ng_trade"]["max_export"] = 1
        elif value > 1:
            extra_fn["ng_trade"]["min_export"] = 1
            extra_fn["ng_trade"]["max_export"] = value
        else:
            extra_fn["ng_trade"]["min_export"] = 0.99
            extra_fn["ng_trade"]["max_export"] = 1.01
        # extra_fn["ng_trade"]["min_export"] = 0
        # extra_fn["ng_trade"]["max_export"] = value
    elif len(exports) > 1:
        raise ValueError("Too many samples for ng_gas_export")
    else:
        extra_fn["ng_trade"]["min_export"] = 1 - 0.01
        extra_fn["ng_trade"]["max_export"] = 1 + 0.01

    ###
    # GSHP capacity constrinats
    ###
    extra_fn["gshp"] = {}
    extra_fn["gshp"]["data"] = pd.read_csv(pop_f)
    gshp_sample = constraints[constraints.attribute == "gshp"].round(5)
    # in sanitize_params we already check that gshp is defined correctly
    # We need to move the res and com gshp capacites together tho.
    gshp_sample = gshp_sample.drop_duplicates(subset="attribute")

    if len(gshp_sample) == 1:
        extra_fn["gshp"]["sample"] = gshp_sample.value.values[0]
    elif len(gshp_sample) > 1:
        raise ValueError("Too many samples for gshp")
    else:
        extra_fn["gshp"]["sample"] = 1

    ###
    # RPS generation target
    ###
    extra_fn["rps"] = {}
    extra_fn["rps"]["data"] = pd.read_csv(rps_f)
    rps_sample = constraints[constraints.attribute == "rps"].round(5)

    if len(rps_sample) == 1:
        extra_fn["rps"]["sample"] = rps_sample.value.values[0]
    elif len(rps_sample) > 1:
        raise ValueError("Too many samples for rps")
    else:
        extra_fn["rps"]["sample"] = 1

    ###
    # CES generation target
    ###
    """Dont include ces as it may not have enforcement mechanisms
    https://eta-publications.lbl.gov/sites/default/files/lbnl_rps_ces_status_report_2024_edition.pdf
    """

    """
    extra_fn["ces"] = {}
    extra_fn["ces"]["data"] = pd.read_csv(ces_f)
    ces_sample = constraints[constraints.attribute == "ces"].round(5)

    if len(ces_sample) == 1:
        extra_fn["ces"]["sample"] = ces_sample.value.values[0]
    elif len(ces_sample) > 1:
        raise ValueError("Too many samples for ces")
    else:
        extra_fn["ces"]["sample"] = 1
    """

    ###
    # TCT Constraint
    ###
    extra_fn["tct"] = {}
    extra_fn["tct"]["data"] = pd.read_csv(tct_f)
    extra_fn["tct"]["sample"] = constraints[constraints.attribute == "tct"].round(5)

    target_names = extra_fn["tct"]["data"].name.to_list()
    sample_names = extra_fn["tct"]["sample"].name.to_list()
    for sample_name in sample_names:
        assert sample_name in target_names

    ###
    # EV Generation Limits
    ###
    extra_fn["ev_gen"] = {}
    extra_fn["ev_gen"]["data"] = pd.read_csv(ev_policy_f, index_col=0)
    extra_fn["ev_gen"]["sample"] = constraints[
        constraints.attribute == "ev_policy"
    ].round(5)

    ###
    # Carbon Limit Constraint
    ###
    extra_fn["co2L"] = {}

    co2_sample = constraints[constraints.attribute == "co2L"].round(5)

    if len(co2_sample) == 1:
        extra_fn["co2L"]["sample"] = co2_sample.value.values[0]
    elif len(co2_sample) > 1:
        raise ValueError("Too many samples for co2L")
    else:
        logger.info("No CO2 Limits provided")
        extra_fn.pop("co2L")

    ###
    # Transmission Expansion Constraint
    ###
    extra_fn["lv"] = {}

    lv_sample = constraints[constraints.attribute == "lv"].round(5)

    if len(lv_sample) == 1:
        # '1 +' because expansion given as per unit
        extra_fn["lv"]["sample"] = 1 + lv_sample.value.values[0]
    elif len(lv_sample) > 1:
        raise ValueError("Too many samples for lv")
    else:
        extra_fn.pop("lv")

    ###
    # Heat Pump cooling constraint
    ###
    extra_fn["hp_cooling"] = True

    ###
    # Loss of Load Probability Constraint
    ###
    extra_fn["lolp"] = True

    n = solve_network(
        n,
        solver_name=solver_name,
        solver_options=solver_opts,
        solving_options=solving_opts,
        log=solving_log,
        extra_fn=extra_fn,
    )

    n.export_to_netcdf(out_network)
