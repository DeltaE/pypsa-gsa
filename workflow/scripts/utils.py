"""Utility Functions"""

from typing import Any
import pandas as pd
import pypsa
from constants import NG_MWH_2_MMCF

import logging
logger = logging.getLogger(__name__)

##################
## GSA Specific ##
##################

def create_salib_problem(parameters: pd.DataFrame) -> dict[str, list[Any]]:
    """Creates SALib problem from scenario configuration."""

    df = parameters.copy()

    problem = {}
    problem["num_vars"] = len(df)
    if problem["num_vars"] <= 1:
        raise ValueError(
            f"Must define at least two variables in problem. User defined "
            f"{problem['num_vars']} variable(s)."
        )

    df["bounds"] = df.apply(lambda row: [row.min_value, row.max_value], axis=1)

    names = df.name.to_list()
    bounds = df.bounds.to_list()
    groups = df.group.to_list()

    problem["names"] = names
    problem["bounds"] = bounds
    problem["groups"] = groups

    num_groups = len(set(groups))
    if num_groups <= 1:
        raise ValueError(
            f"Must define at least two groups in problem. User defined "
            f"{num_groups} group(s)."
        )

    return problem

#####################
## General Helpers ##
#####################

def calculate_annuity(lifetime: int, dr: float | int):
    """
    Calculate the annuity factor for an asset. 
    """
    if dr > 0:
        return dr / (1.0 - 1.0 / (1.0 + dr) ** lifetime)
    else:
        return 1 / lifetime
    
#################
## Constraints ##
#################

"""
To calcualte the scaled EE, we need the actual values of the RHS. This is complicated, 
as the values are calculated in the extra functionality of solve. This makes it hard to 
write out these values to append to the sample.

Therefore, this module constains functions that are used both in apply_sample and solve
to process the RHS value. The calculations are seperate (ie. solve network still reads 
in the unscaled values). 
"""


def get_region_buses(n: pypsa.Network, region_list: list[str]) -> pd.DataFrame:
    """Filters buses based on regional input."""
    return n.buses[
        (
            n.buses.country.isin(region_list)
            | n.buses.reeds_zone.isin(region_list)
            | n.buses.reeds_state.isin(region_list)
            | n.buses.interconnect.str.lower().isin(region_list)
            | n.buses.nerc_reg.isin(region_list)
            | (1 if "all" in region_list else 0)
        )
    ]
    
###
# Transmission Expansion 
###


def get_existing_lv(n: pypsa.Network) -> float:
    """Gets exisitng line volume."""
    ac_links_existing = n.links.carrier == "AC" if not n.links.empty else pd.Series()
    return n.links.loc[ac_links_existing, "p_nom"] @ n.links.loc[ac_links_existing, "length"]


###
# RPS and CES 
###

def concat_rps_standards(n: pypsa.Network, rps: pd.DataFrame) -> pd.DataFrame:
    
    planning_horizon = n.investment_periods[0]

    # Concatenate all portfolio standards
    portfolio_standards = rps.copy()
    portfolio_standards["pct"] = portfolio_standards.pct.clip(upper=1)
    portfolio_standards = portfolio_standards[
        (portfolio_standards.pct > 0.0)
        & (portfolio_standards.planning_horizon == planning_horizon)
        & (portfolio_standards.region.isin(n.buses.reeds_state.unique()))
    ]
    
    return portfolio_standards
    

def get_rps_eligible(n: pypsa.Network, rps_region: str, rps_carrier: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Buses and generators that can contribute to RPS in PyPSA format."""
    
    region_list = [region.strip() for region in rps_region.split(",")]
    region_buses = get_region_buses(n, region_list)

    if region_buses.empty:
        return pd.DataFrame(), pd.DataFrame()

    carriers = [carrier.strip() for carrier in rps_carrier.split(",")]
    carriers.append("load")

    # Filter region generators
    region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
    region_gens = region_gens[region_gens.carrier.isin(carriers)]
    
    return region_buses, region_gens
    
def get_rps_generation(n: pypsa.Network, planning_horizon: int, region_gens: pd.DataFrame):
    """LHS of constrint. Generators that can contribute to RPS. Returns linopy sum."""
    p_eligible = n.model["Generator-p"].sel(
        period=planning_horizon,
        Generator=region_gens.index,
    )
    return p_eligible.sum()

def get_rps_demand_gsa(n: pypsa.Network, planning_horizon: int, region_buses: pd.DataFrame):
    """Demand to benchmark the rps against.
    
    WARNING! This is an aproximation for GSA purposes only. The real constraint has a RHS
    of zero and sums actual outgoing flows. 
    """
    load_buses = n.loads
    load_buses["country"] = load_buses.bus.map(n.buses.country)
    load_buses = load_buses[
        (load_buses.country.isin(region_buses.index))
        & (load_buses.carrier.str.contains("-elec"))
    ]

    return n.loads_t.p_set.loc[planning_horizon, load_buses.index].sum().sum()

def get_rps_demand_actual(n: pypsa.Network, planning_horizon: int, region_buses: pd.DataFrame):
    """LHS of constrint. Returns linopy sum.
    
    This is the sum of outflowing electricity from power sector links. 
    """
    # power level buses
    pwr_buses = n.buses[(n.buses.carrier == "AC") & (n.buses.index.isin(region_buses.index))]
    # links delievering power within the region; removes any transmission links
    pwr_links = n.links[(n.links.bus0.isin(pwr_buses.index)) & ~(n.links.bus1.isin(pwr_buses.index))]
    region_demand = n.model["Link-p"].sel(period=planning_horizon, Link=pwr_links.index)
    
    return region_demand.sum()

###
# Natural Gas Trade
###

def format_raw_ng_trade_data(prod: pd.DataFrame, link_suffix: str | None = None) -> pd.DataFrame:
    """Formats the raw EIA natural gas trade data into something usable"""

    def _format_link_name(s: str) -> str:
        states = s.split("-")
        return f"{states[0]} {states[1]} gas"

    def _format_data(
        prod: pd.DataFrame,
        link_suffix: str | None = None,
    ) -> pd.DataFrame:
        df = prod.copy()
        df["link"] = df.state.map(_format_link_name)
        if link_suffix:
            df["link"] = df.link + link_suffix

        # convert mmcf to MWh
        df["value"] = df["value"] * NG_MWH_2_MMCF

        return df[["link", "value"]].rename(columns={"value": "rhs"}).set_index("link")
    
    return _format_data(prod, link_suffix)

def get_ng_trade_links(n: pypsa.Network, direction: str) -> list[str]: 
    """Gets natural gas trade links within the network."""
    
    assert direction in ("imports", "exports")
    
    if direction == "imports":
        return n.links[
            (n.links.carrier == "gas trade") & (n.links.bus0.str.endswith(" gas trade"))
        ].index.to_list()
    elif direction == "exports":
        return n.links[
            (n.links.carrier == "gas trade") & (n.links.bus0.str.endswith(" gas"))
        ].index.to_list()
    else: 
        raise ValueError(f"Undefined control flow for direction {direction}")


###
# Urban Rural fraction for GSHP
###


def get_urban_rural_fraction(pop: pd.DataFrame) -> pd.DataFrame:
    """Gets urban rural fraction for the GSHP capacity constraint."""

    pop["urban_rural_fraction"] = (pop.urban_fraction / pop.rural_fraction).round(5)
    return pop.set_index("name")["urban_rural_fraction"].to_dict()