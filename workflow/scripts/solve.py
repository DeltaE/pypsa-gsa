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
    get_region_buses,
    get_rps_demand_actual,
    get_rps_eligible,
    get_rps_generation,
    concat_rps_standards,
    format_raw_ng_trade_data,
    get_ng_trade_links,
    get_urban_rural_fraction,
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
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}",
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

def add_technology_capacity_target_constraints(
    n: pypsa.Network, data: pd.DataFrame, sample: pd.DataFrame
):
    """
    Add Technology Capacity Target (TCT) constraint to the network.

    Add minimum or maximum levels of generator nominal capacity per carrier for individual regions.
    Each constraint can be designated for a specified planning horizon in multi-period models.
    Opts and path for technology_capacity_targets.csv must be defined in config.yaml.
    Default file is available at config/policy_constraints/technology_capacity_targets.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-TCT-24H]
    electricity:
        technology_capacity_target: config/policy_constraints/technology_capacity_target.csv
    """

    tct_data = data.copy()

    # apply sample
    tct_data = tct_data.set_index("name")
    for name, sample_value in zip(sample.name, sample.value):
        tct_data.loc[name, "max"] *= sample_value
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
            lhs_s = n.model["StorageUnit-p_nom"].loc[lhs_storage_ext.index].groupby(grouper_s).sum()
        else:
            lhs_s = None

        if not lhs_link_ext.empty:
            grouper_l = pd.concat(
                [lhs_link_ext.bus.map(n.buses.country), lhs_link_ext.carrier],
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
            assert (
                target["max"] >= lhs_existing
            ), f"TCT constraint of {target['max']} MW for {target['carrier']} must be at least {lhs_existing}"

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
    """

    if rps.empty:
        return
    
    portfolio_standards = concat_rps_standards(n, rps)

    # Iterate through constraints
    for _, constraint_row in portfolio_standards.iterrows():
        
        region_buses, region_gens = get_rps_eligible(n, constraint_row.region, constraint_row.carrier)
        
        if region_buses.empty:
            continue

        if not region_gens.empty:
            region_demand = get_rps_demand_actual(n, constraint_row.planning_horizon, region_buses)
            region_gen = get_rps_generation(n, constraint_row.planning_horizon, region_gens)

            lhs = region_gen - constraint_row.pct * region_demand * sample
            rhs = 0

            # Add constraint
            n.model.add_constraints(
                lhs >= rhs,
                name=f"GlobalConstraint-{constraint_row.name}_{constraint_row.planning_horizon}_{policy_name}_limit",
            )
            logger.info(
                f"Added RPS {constraint_row.region} for {constraint_row.planning_horizon}.",
            )


def add_sector_co2_constraints(n: pypsa.Network, sample: float):
    """
    Adds sector co2 constraints.

    Parameters
    ----------
        n : pypsa.Network
        config : dict
    """

    def apply_national_limit(
        n: pypsa.Network, year: int, value: float, sector: str | None = None
    ):
        """For every snapshot, sum of co2 and ch4 must be less than limit."""
        if sector:
            stores = n.stores[
                (
                    (n.stores.index.str.endswith(f"{sector}-co2"))
                    | (n.stores.index.str.endswith(f"{sector}-ch4"))
                )
            ].index
            name = f"co2_limit-{year}-{sector}"
            log_statement = f"Adding national {sector} co2 Limit in {year} of"
        else:
            stores = n.stores[
                (
                    (n.stores.index.str.endswith("-co2"))
                    | (n.stores.index.str.endswith("-ch4"))
                )
            ].index
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
        logger.warning("No co2 policies applied")
        return

    sectors = df.sector.unique()

    for sector in sectors:
        df_sector = df[df.sector == sector]
        states = df_sector.state.unique()

        for state in states:
            df_state = df_sector[df_sector.state == state]
            years = [x for x in df_state.year.unique() if x in n.investment_periods]

            if not years:
                logger.warning(
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
                        apply_national_limit(n, year, value)
                    else:
                        apply_national_limit(n, year, value, sector)
                else:
                    raise ValueError(state.lower())

def add_ng_import_export_limits(
    n: pypsa.Network, ng_trade: dict[str, pd.DataFrame], limits: dict[str, float]
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
                    # logger.warning(f"Can not set gas import limit for {link}")
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
                    # logger.warning(f"Can not set gas import limit for {link}")
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

    import_min = limits.get("import_min", 1)
    import_max = limits.get("import_max", 1)
    export_min = limits.get("export_min", 1)
    export_max = limits.get("export_max", 1)

    # to avoid numerical issues, ensure there is a gap between min/max constraints
    if abs(import_max - import_min) < 0.0001:
        import_min -= 0.001
        import_max += 0.001
        if import_min < 0:
            import_min = 0

    if abs(export_max - export_min) < 0.0001:
        export_min -= 0.001
        export_max += 0.001
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

        lhs = n.model["Link-p_nom"].loc[heating_hps] - n.model["Link-p_nom"].loc[cooling_hps]
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
        assert len(sample_mode) == 1

        evs = n.links[n.links.carrier == f"trn-elec-veh-{mode_mapper[mode]}"].index
        dem_names = n.loads[n.loads.carrier == f"trn-veh-{mode_mapper[mode]}"].index
        dem = n.loads_t["p_set"][dem_names]

        for investment_period in n.investment_periods:
            ratio = policy.at[investment_period, mode] / 100  # input is percentage
            eff = n.links.loc[evs].efficiency.mean()
            lhs = n.model["Link-p"].loc[investment_period].sel(Link=evs).sum()
            rhs_ref = dem.loc[investment_period].sum().sum() * ratio / eff
            rhs = rhs_ref + rhs_ref * sample_mode.value.values[0]

            n.model.add_constraints(
                lhs <= rhs, name=f"Link-ev_gen_{mode}_{investment_period}"
            )

def extra_functionality(n, sns):
    """
    Collects supplementary constraints which will be passed to `pypsa.optimization.optimize`
    """

    opts = n.extra_fn

    # if "rps" in opts:
    #     add_RPS_constraints(n, "rps", opts["rps"]["data"], opts["rps"]["sample"])
    # if "ces" in opts:
    #     add_RPS_constraints(n, "ces", opts["ces"]["data"], opts["ces"]["sample"])
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
    if "ng_limits" in opts:
        add_ng_import_export_limits(n, opts["ng_limits"])
    if "lv" in opts:
        add_transmission_limit(n, opts["lv"]["sample"])
    if "hp_cooling" in opts:
        add_cooling_heat_pump_constraints(n)


###
# Prepare Network
###


def prepare_network(
    n,
    clip_p_max_pu: Optional[bool | float] = None,
    noisy_costs: Optional[bool] = None,
    foresight: Optional[str] = None,
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
        solving_log = snakemake.log.solver
        out_network = snakemake.output.network
        pop_f = snakemake.input.pop_layout_f
        ng_dommestic_f = snakemake.input.ng_domestic_f
        ng_international_f = snakemake.input.ng_international_f
        rps_f = snakemake.input.rps_f
        ces_f = snakemake.input.ces_f
        tct_f = snakemake.input.tct_f
        ev_policy_f = snakemake.input.ev_policy_f
        constraints_meta = snakemake.input.constraints
    else:
        in_network = "results/EvPolicy/gsa/modelruns/0/n.nc"
        solver_name = "gurobi"
        solving_opts_config = "config/solving.yaml"
        solving_log = ""
        out_network = ""
        pop_f = "results/EvPolicy/gsa/constraints/pop_layout.csv"
        ng_dommestic_f = "results/EvPolicy/gsa/constraints/ng_domestic.csv"
        ng_international_f = "results/EvPolicy/gsa/constraints/ng_international.csv"
        rps_f = "results/EvPolicy/gsa/constraints/rps.csv"
        ces_f = "results/EvPolicy/gsa/constraints/ces.csv"
        tct_f = "results/EvPolicy/gsa/constraints/tct.csv"
        ev_policy_f = "results/EvPolicy/gsa/constraints/ev_policy.csv"
        constraints_meta = "results/EvPolicy/gsa/modelruns/0/constraints.csv"

        with open(solving_opts_config, "r") as f:
            solving_opts_all = yaml.safe_load(f)

        solving_opts = solving_opts_all["solving"]["options"]
        solver_opts = solving_opts_all["solving"]["solver_options"]["gurobi-default"]

    n = pypsa.Network(in_network)

    # for land use constraint
    solving_opts["foresight"] = "perfect"

    np.random.seed(solving_opts.get("seed", 123))

    n = prepare_network(n, **solving_opts)

    # holds sampled data that needs to be applied to RHS of constraints
    constraints = pd.read_csv(constraints_meta)

    extra_fn = {}

    ###
    # natural gas constraints
    ###
    extra_fn["ng_trade"] = {}
    extra_fn["ng_trade"]["domestic"] = pd.read_csv(ng_dommestic_f, index_col=0)
    extra_fn["ng_trade"]["international"] = pd.read_csv(ng_international_f, index_col=0)
    imports = constraints[constraints.attribute == "nat_gas_import"].round(5)
    exports = constraints[constraints.attribute == "nat_gas_export"].round(5)

    if len(imports) == 1:
        extra_fn["ng_trade"]["min_import"] = imports.value.values[0]
        extra_fn["ng_trade"]["max_import"] = imports.value.values[0]
    elif len(imports) > 1:
        raise ValueError("Too many samples for ng_gas_import")
    else:
        extra_fn["ng_trade"]["min_import"] = 1
        extra_fn["ng_trade"]["max_import"] = 1

    if len(exports) == 1:
        extra_fn["ng_trade"]["min_export"] = exports.value.values[0]
        extra_fn["ng_trade"]["max_export"] = exports.value.values[0]
    elif len(exports) > 1:
        raise ValueError("Too many samples for ng_gas_export")
    else:
        extra_fn["ng_trade"]["min_export"] = 1
        extra_fn["ng_trade"]["max_export"] = 1

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
    extra_fn["ces"] = {}
    extra_fn["ces"]["data"] = pd.read_csv(ces_f)
    ces_sample = constraints[constraints.attribute == "ces"].round(5)

    if len(ces_sample) == 1:
        extra_fn["ces"]["sample"] = ces_sample.value.values[0]
    elif len(ces_sample) > 1:
        raise ValueError("Too many samples for ces")
    else:
        extra_fn["ces"]["sample"] = 1

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
        logger.warning("No CO2 Limits provided")
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


    n = solve_network(
        n,
        solver_name=solver_name,
        solver_options=solver_opts,
        solving_options=solving_opts,
        log=solving_log,
        extra_fn=extra_fn,
    )

    n.export_to_netcdf(out_network)
