"""
Modified solve network file from PyPSA-USA

https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/solve_network.py
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import pypsa

from constants import CONVENTIONAL_CARRIERS

NG_MWH_2_MMCF = 305

from typing import Optional
import yaml


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


def add_RPS_constraints(n: pypsa.Network, policy: pd.DataFrame):
    """
    Add Renewable Portfolio Standards constraint to the network.

    Add percent levels of generator production (MWh) per carrier or groups of
    carriers for individual countries. Each constraint can be designated for a
    specified planning horizon in multi-period models.

    Parameters
    ----------
    n : pypsa.Network
    config : dict
    """

    df = policy.copy()

    for _, pct_lim in df.iterrows():
        regions = [r.strip() for r in pct_lim.region.split(",")]
        buses = n.buses[
            (
                n.buses.country.isin(regions)
                | n.buses.reeds_state.isin(regions)
                | n.buses.interconnect.str.lower().isin(regions)
                | (1 if "all" in regions else 0)
            )
        ]

        if buses.empty:
            continue

        carriers = [c.strip() for c in pct_lim.carrier.split(",")]

        # generators
        region_gens = n.generators[n.generators.bus.isin(buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if not region_gens.empty:
            p_eligible = n.model["Generator-p"].sel(
                period=pct_lim.planning_horizon,
                Generator=region_gens_eligible.index,
            )
            lhs = p_eligible.sum()

            region_demand = (
                n.loads_t.p_set.loc[
                    pct_lim.planning_horizon,
                    n.loads.bus.isin(buses.index),
                ]
                .sum()
                .sum()
            )

            rhs = pct_lim.pct * region_demand

            n.model.add_constraints(
                lhs >= rhs,
                name=f"GlobalConstraint-{pct_lim.name}_{pct_lim.planning_horizon}_rps_limit",
            )
            logger.info(
                f"Adding RPS {pct_lim.name}_{pct_lim.planning_horizon} for {pct_lim.planning_horizon}.",
            )


def add_interface_limits(n, itl: pd.DataFrame, transport: bool = True):
    """
    Adds interface transmission limits to constrain inter-regional transfer
    capacities based on user-defined inter-regional transfer capacity limits.
    """

    df = itl.copy()

    for _, interface in df.iterrows():
        regions_list_r = [region.strip() for region in interface.r.split(",")]
        regions_list_rr = [region.strip() for region in interface.rr.split(",")]

        zone0_buses = n.buses[n.buses.country.isin(regions_list_r)]
        zone1_buses = n.buses[n.buses.country.isin(regions_list_rr)]
        if zone0_buses.empty | zone1_buses.empty:
            continue

        logger.info(f"Adding Interface Transmission Limit for {interface.interface}")

        interface_lines_b0 = n.lines[
            n.lines.bus0.isin(zone0_buses.index) & n.lines.bus1.isin(zone1_buses.index)
        ]
        interface_lines_b1 = n.lines[
            n.lines.bus0.isin(zone1_buses.index) & n.lines.bus1.isin(zone0_buses.index)
        ]
        interface_links_b0 = n.links[
            n.links.bus0.isin(zone0_buses.index) & n.links.bus1.isin(zone1_buses.index)
        ]
        interface_links_b1 = n.links[
            n.links.bus0.isin(zone1_buses.index) & n.links.bus1.isin(zone0_buses.index)
        ]

        if not n.lines.empty:
            line_flows = n.model["Line-s"].loc[:, interface_lines_b1.index].sum(
                dim="Line",
            ) - n.model["Line-s"].loc[:, interface_lines_b0.index].sum(dim="Line")
        else:
            line_flows = 0.0
        lhs = line_flows

        interface_links = pd.concat([interface_links_b0, interface_links_b1])

        # Apply link constraints if RESOLVE constraint or if zonal model.
        # ITLs should usually only apply to AC lines if DC PF is used.
        if not (interface_links.empty) and (
            "RESOLVE" in interface.interface or transport
        ):
            link_flows = n.model["Link-p"].loc[:, interface_links_b1.index].sum(
                dim="Link",
            ) - n.model["Link-p"].loc[:, interface_links_b0.index].sum(dim="Link")
            lhs += link_flows

        rhs_pos = interface.MW_f0 * -1
        n.model.add_constraints(lhs >= rhs_pos, name=f"ITL_{interface.interface}_pos")

        rhs_neg = interface.MW_r0
        n.model.add_constraints(lhs <= rhs_neg, name=f"ITL_{interface.interface}_neg")


def add_SAFER_constraints(n: pypsa.Network, safer: pd.DataFrame):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand
    for regions defined in configuration file. Renewable generators and storage
    do not contribute towards PRM.

    Parameters
    ----------
        n : pypsa.Network
        config : dict
    """

    df = safer.copy()

    for _, prm in df.iterrows():
        region_list = [region_.strip() for region_ in prm.region.split(",")]
        region_buses = n.buses[
            (
                n.buses.country.isin(region_list)
                | n.buses.reeds_state.isin(region_list)
                | n.buses.interconnect.str.lower().isin(region_list)
                | n.buses.nerc_reg.isin(region_list)
                | (1 if "all" in region_list else 0)
            )
        ]

        if region_buses.empty:
            continue

        peakdemand = (
            n.loads_t.p_set.loc[
                prm.planning_horizon,
                n.loads.bus.isin(region_buses.index),
            ]
            .sum(axis=1)
            .max()
        )
        margin = 1.0 + prm.prm
        planning_reserve = peakdemand * margin

        conventional_carriers = CONVENTIONAL_CARRIERS

        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        ext_gens_i = region_gens.query(
            "carrier in @conventional_carriers & p_nom_extendable",
        ).index

        p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
        lhs = p_nom.sum()
        exist_conv_caps = region_gens.query(
            "~p_nom_extendable & carrier in @conventional_carriers",
        ).p_nom.sum()
        rhs = planning_reserve - exist_conv_caps
        n.model.add_constraints(
            lhs >= rhs,
            name=f"GlobalConstraint-{prm.name}_{prm.planning_horizon}_PRM",
        )


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_sector_co2_constraints(n: pypsa.Network, co2L: pd.DataFrame):
    """
    Adds sector co2 constraints.

    Parameters
    ----------
        n : pypsa.Network
        config : dict
    """

    def apply_total_state_limit(n, year, state, value):

        sns = n.snapshots
        snapshot = sns[sns.get_level_values("period") == year][-1]

        stores = n.stores[
            (n.stores.index.str.startswith(state))
            & (n.stores.index.str.endswith("-co2"))
        ].index

        lhs = n.model["Store-e"].loc[snapshot, stores].sum()

        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=f"co2_limit-{year}-{state}")

        logger.info(
            f"Adding {state} co2 Limit in {year} of {rhs* 1e-6} MMT CO2",
        )

    def apply_sector_state_limit(n, year, state, sector, value):

        sns = n.snapshots
        snapshot = sns[sns.get_level_values("period") == year][-1]

        stores = n.stores[
            (n.stores.index.str.startswith(state))
            & (n.stores.index.str.endswith(f"{sector}-co2"))
        ].index

        lhs = n.model["Store-e"].loc[snapshot, stores].sum()

        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=f"co2_limit-{year}-{state}-{sector}")

        logger.info(
            f"Adding {state} co2 Limit for {sector} in {year} of {rhs* 1e-6} MMT CO2",
        )

    def apply_total_national_limit(n, year, value):

        sns = n.snapshots
        snapshot = sns[sns.get_level_values("period") == year][-1]

        stores = n.stores[n.stores.index.str.endswith("-co2")].index

        lhs = n.model["Store-e"].loc[snapshot, stores].sum()

        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=f"co2_limit-{year}")

        logger.info(
            f"Adding national co2 Limit in {year} of {rhs* 1e-6} MMT CO2",
        )

    def apply_sector_national_limit(n, year, sector, value):

        sns = n.snapshots
        snapshot = sns[sns.get_level_values("period") == year][-1]

        stores = n.stores[n.stores.index.str.endswith(f"{sector}-co2")].index

        lhs = n.model["Store-e"].loc[snapshot, stores].sum()

        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=f"co2_limit-{year}-{sector}")

        logger.info(
            f"Adding national co2 Limit for {sector} sector in {year} of {rhs* 1e-6} MMT CO2",
        )

    df = co2L.copy()

    sectors = df.sector.unique()

    for sector in sectors:

        df_sector = df[df.sector == sector]
        states = df_sector.state.unique()

        for state in states:

            df_state = df_sector[df_sector.state == state]
            years = [x for x in df_state.year.unique() if x in n.investment_periods]

            if not years:
                logger.warning(f"No co2 policies applied for {sector} in {year}")
                continue

            for year in years:

                df_limit = df_state[df_state.year == year].reset_index(drop=True)
                assert df_limit.shape[0] == 1

                # results calcualted in T CO2, policy given in MMT CO2
                value = df_limit.loc[0, "co2_limit_mmt"] * 1e6

                if state.upper() == "USA":

                    if sector == "all":
                        apply_total_national_limit(n, year, value)
                    else:
                        apply_sector_national_limit(n, year, sector, value)

                else:

                    if sector == "all":
                        apply_total_state_limit(n, year, state, value)
                    else:
                        apply_sector_state_limit(n, year, state, sector, value)


def add_ng_import_export_limits(n: pypsa.Network, ng_trade: dict[str, pd.DataFrame]):

    def _format_link_name(s: str) -> str:
        states = s.split("-")
        return f"{states[0]} {states[1]} gas"

    def _format_domestic_data(
        prod: pd.DataFrame,
        link_suffix: Optional[str] = None,
    ) -> pd.DataFrame:

        df = prod.copy()
        df["link"] = df.state.map(_format_link_name)
        if link_suffix:
            df["link"] = df.link + link_suffix

        # convert mmcf to MWh
        df["value"] = df["value"] * 1000 / NG_MWH_2_MMCF

        return df[["link", "value"]].rename(columns={"value": "rhs"}).set_index("link")

    def _format_international_data(
        prod: pd.DataFrame,
        link_suffix: Optional[str] = None,
    ) -> pd.DataFrame:

        df = prod.copy()
        df = df[["value", "state"]].groupby("state", as_index=False).sum()
        df = df[~(df.state == "USA")].copy()

        df["link"] = df.state.map(_format_link_name)
        if link_suffix:
            df["link"] = df.link + link_suffix

        # convert mmcf to MWh
        df["value"] = df["value"] * 1000 / NG_MWH_2_MMCF

        return df[["link", "value"]].rename(columns={"value": "rhs"}).set_index("link")

    def add_import_limits(n, imports):
        """
        Sets gas import limit over each year.
        """

        weights = n.snapshot_weightings.objective

        links = n.links[n.links.carrier.str.endswith("gas import")].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = imports.at[link, "rhs"]
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                n.model.add_constraints(lhs <= rhs, name=f"ng_limit-{year}-{link}")

    def add_export_limits(n, exports):
        """
        Sets maximum export limit over the year.
        """

        weights = n.snapshot_weightings.objective

        links = n.links[n.links.carrier.str.endswith("gas export")].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = exports.at[link, "rhs"]
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                n.model.add_constraints(lhs >= rhs, name=f"ng_limit-{year}-{link}")

    
    dom_imports = ng_trade["dom_imports"].copy()
    dom_exports = ng_trade["dom_exports"].copy()
    int_imports = ng_trade["int_imports"].copy()
    int_exports = ng_trade["int_exports"].copy()

    # add domestic limits

    imports = _format_domestic_data(dom_imports, " import")
    exports = _format_domestic_data(dom_exports, " export")

    # add_import_limits(n, imports)
    add_export_limits(n, exports)

    # add international limits

    imports = _format_international_data(int_imports, " import")
    exports = _format_international_data(int_exports, " export")

    # add_import_limits(n, imports)
    add_export_limits(n, exports)


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


def add_gshp_capacity_constraint(n: pypsa.Network, pop_layout: pd.DataFrame):
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
    
    df["urban_rural_fraction"] = (df.urban_fraction / df.rural_fraction).round(2)
    fraction = df.set_index("name")["urban_rural_fraction"].to_dict()

    ashp = n.links[n.links.index.str.endswith("ashp")].copy()
    gshp = n.links[n.links.index.str.endswith("gshp")].copy()
    if gshp.empty:
        return

    assert len(ashp) == len(gshp)

    gshp["urban_rural_fraction"] = gshp.bus0.map(fraction)

    ashp_capacity = n.model["Link-p_nom"].loc[ashp.index]
    gshp_capacity = n.model["Link-p_nom"].loc[gshp.index]
    gshp_multiplier = gshp["urban_rural_fraction"]

    lhs = ashp_capacity - gshp_capacity.mul(gshp_multiplier.values)
    rhs = 0

    n.model.add_constraints(lhs >= rhs, name=f"Link-gshp_capacity_ratio")


def extra_functionality(n, sns):
    """
    Collects supplementary constraints which will be passed to
    `pypsa.optimization.optimize`
    """

    opts = n.extra_fn

    if "rps" in opts and n.generators.p_nom_extendable.any():
        add_RPS_constraints(n, opts["rps"])
    if "safer" in opts and n.generators.p_nom_extendable.any():
        add_SAFER_constraints(n, opts["safer"])
    if "itl" in opts:
        transport = True
        add_interface_limits(n, opts["itl"], transport)
    if "co2L" in opts:
        add_sector_co2_constraints(n, opts["co2L"])
    if "gshp" in opts:
        add_gshp_capacity_constraint(n, opts["gshp"])
    if "ng_limits" in opts:
        add_ng_import_export_limits(n, opts["ng_limits"])
    if "hp_cooling" in opts:
        add_cooling_heat_pump_constraints(n, opts)

    add_battery_constraints(n)


###
# Prepare Network
###


def prepare_network(
    n,
    clip_p_max_pu: Optional[bool | float] = None,
    load_shedding: Optional[bool | float] = None,
    noisy_costs: Optional[bool] = None,
    foresight: Optional[str] = None,
    **kwargs,
):

    if clip_p_max_pu:
        if isinstance(clip_p_max_pu, float):
            _clip_p_max(n, clip_p_max_pu)
        else:
            _clip_p_max(n)

    if load_shedding:
        if isinstance(load_shedding, float):
            _apply_load_shedding(n, load_shedding)
        else:
            _apply_load_shedding(n)

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


def _apply_load_shedding(n: pypsa.Network, value: Optional[float] = None) -> None:
    """Intersect between macroeconomic and surveybased willingness to pay

    http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
    """

    # TODO: retrieve color and nice name from config
    n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")

    buses_i = n.buses.query("carrier == 'AC'").index

    if not value:
        value = 100

    n.madd(
        "Generator",
        buses_i,
        " load",
        bus=buses_i,
        carrier="load",
        sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
        marginal_cost=value,  # Eur/kWh
        p_nom=1e9,  # kW
    )


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
        pypsa_usa_opts = snakemake.params.pypsa_usa_opts
        solving_log = snakemake.log.solver
        out_network = snakemake.output.network
        # extra constraints
        itl_f = snakemake.input.itl
        safer_f = snakemake.input.safer
        rps_f = snakemake.input.rps
        co2L_f = snakemake.input.co2L
        # pypsa-usa specific 
        pop_f = snakemake.input.pop_layout
        ng_dom_imports = snakemake.input.ng_domestic_imports
        ng_dom_exports = snakemake.input.ng_domestic_exports
        ng_int_imports = snakemake.input.ng_international_imports
        ng_int_exports = snakemake.input.ng_international_exports
    else:
        in_network = "results/Western/modelruns/0/n.nc"
        solver_name = "gurobi"
        solving_opts_config = "config/solving.yaml"
        solving_log = ""
        out_network = ""
        pypsa_usa_opts = {"ng_limits": True, "hp_capacity": True, "hp_cooling": True}
        # extra constraints
        itl_f = "config/constraints/itl.csv"
        safer_f = ""
        rps_f = "config/constraints/rps.csv"
        co2L_f = "config/constraints/co2L.csv"
        # pypsa-usa specific 
        pop_f = "config/pypsa-usa/pop_layout_elec_s33_c4m.csv"
        ng_dom_imports = "config/pypsa-usa/domestic_imports.csv"
        ng_dom_exports = "config/pypsa-usa/domestic_exports.csv"
        ng_int_imports = "config/pypsa-usa/international_imports.csv"
        ng_int_exports = "config/pypsa-usa/international_exports.csv"

        with open(solving_opts_config, "r") as f:
            solving_opts_all = yaml.safe_load(f)

        solving_opts = solving_opts_all["solving"]["options"]
        solver_opts = solving_opts_all["solving"]["solver_options"]["gurobi-default"]

    n = pypsa.Network(in_network)

    # for land use constraint
    solving_opts["foresight"] = "perfect"

    np.random.seed(solving_opts.get("seed", 123))

    n = prepare_network(n, **solving_opts)

    extra_fn = {}
    if itl_f:
        extra_fn["itl"] = pd.read_csv(itl_f)
    if safer_f:
        extra_fn["safer"] = pd.read_csv(safer_f)
    if rps_f:
        extra_fn["rps"] = pd.read_csv(rps_f)
    if co2L_f:
        extra_fn["co2L"] = pd.read_csv(co2L_f)
    if pypsa_usa_opts["ng_limit"]:
        extra_fn["ng_limits"] = {}
        extra_fn["ng_limits"]["dom_imports"] = pd.read_csv(ng_dom_imports, index_col=0)
        extra_fn["ng_limits"]["dom_exports"] = pd.read_csv(ng_dom_exports, index_col=0)
        extra_fn["ng_limits"]["int_imports"] = pd.read_csv(ng_int_imports, index_col=0)
        extra_fn["ng_limits"]["int_exports"] = pd.read_csv(ng_int_exports, index_col=0)
    if pypsa_usa_opts["hp_capacity"]:
        extra_fn["gshp"] = pd.read_csv(pop_f)
    if pypsa_usa_opts["hp_cooling"]:
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
