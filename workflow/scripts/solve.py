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
        solving_log = snakemake.log.solver
        out_network = snakemake.output.network
        # extra constraints
        itl_f = snakemake.input.itl
        safer_f = snakemake.input.safer
        rps_f = snakemake.input.rps
        co2L_f = snakemake.input.co2L
    else:
        in_network = "results/Western/modelruns/0/n.nc"
        solver_name = "gurobi"
        solving_opts_config = "config/solving.yaml"
        solving_log = ""
        out_network = ""
        # extra constraints
        itl_f = "results/Western/constraints/itl.csv"
        safer_f = ""
        rps_f = "results/Western/constraints/rps.csv"
        co2L_f = "results/Western/constraints/co2L.csv"

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

    n = solve_network(
        n,
        solver_name=solver_name,
        solver_options=solver_opts,
        solving_options=solving_opts,
        log=solving_log,
        extra_fn=extra_fn,
    )

    n.export_to_netcdf(out_network)
