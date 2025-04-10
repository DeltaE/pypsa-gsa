"""Applies sampled value to the base network"""

import pandas as pd
import numpy as np
import pypsa
from typing import Any
from pathlib import Path
import yaml
from dataclasses import dataclass
from utils import calculate_annuity
from utils import (
    get_existing_lv,
    get_rps_demand_gsa,
    concat_rps_standards,
    get_rps_eligible,
    get_ng_trade_links,
    format_raw_ng_trade_data,
    get_urban_rural_fraction,
    configure_logging
)
from constants import CACHED_ATTRS, CONSTRAINT_ATTRS

import logging
logger = logging.getLogger(__name__)


@dataclass
class CapitalCostCache:
    """Perform intermediate capital cost calculation."""

    component: str
    capital_cost: float = None
    lifetime: int = None
    discount_rate: float = None
    fixed_cost: float = None
    occ: float = None
    itc: float = None
    vmt_per_year: float = None

    def is_valid_data(self) -> bool:
        """Checks that all data is present before applying sample"""
        if self.capital_cost:
            return True
        if not self.occ:
            raise ValueError("occ")
        elif not self.fixed_cost:
            raise ValueError("fom")
        elif not self.discount_rate:
            raise ValueError("discount rate")
        elif not (self.lifetime or self.vmt_per_year):
            raise ValueError("lifetime or vmt_per_year")
        else:
            return True

    def calculate_capex(self, transport: bool = False) -> float:
        """Capex is an intermediate calcualtion.

        Fixed cost is given in same units as occ.
        """
        assert self.is_valid_data()
        if self.capital_cost:
            logger.info("Returning pre-defined capital cost.")
            return round(self.capital_cost, 5)
        elif transport:
            return self._calc_transport_capex()
        else:
            return self._calc_standard_capex()

    def _calc_standard_capex(self) -> float:
        assert self.discount_rate < 1  # ensures is per_unit
        annuity = calculate_annuity(self.lifetime, self.discount_rate)
        capex = (self.occ + self.fixed_cost) * annuity

        if self.itc:
            assert self.itc < 1  # ensures is per_unit
            return round(capex * (1 - self.itc), 5)
        else:
            return round(capex, 5)

    def _calc_transport_capex(self) -> float:
        """OCC comes as 'usd' and needs to be converted to usd/kvmt."""
        assert self.discount_rate < 1  # ensures is per_unit
        annuity = calculate_annuity(self.lifetime, self.discount_rate)

        assert self.vmt_per_year
        capex = ((self.occ / self.vmt_per_year) + self.fixed_cost) * annuity

        return round(capex, 5)

@dataclass
class MethaneLeakageCache:
    """Perform intermediate methane leakage calculation."""

    component: str
    gwp: float = None
    leakage: float = None

    def is_valid_data(self) -> bool:
        """Checks that all data is present before applying sample"""
        if not self.gwp:
            raise ValueError("gwp")
        elif not self.leakage:
            raise ValueError("leakage")
        else:
            return True

    def calculate_leakage(self) -> float:
        assert self.leakage < 1  # confirm per_unit
        return round(self.gwp * self.leakage, 5)

def is_valid_carrier(n: pypsa.Network, params: pd.DataFrame) -> bool:
    """Check all defined carriers are in the network."""

    df = params.copy()

    sa_cars = df.carrier.unique()
    n_cars = n.carriers.index.to_list()
    n_cars.append("portfolio")  # for aggregating constraints
    n_cars.extend(["leakage_upstream", "leakage_downstream"])  # ng leaks

    errors = []

    for cars in sa_cars:
        if ";" in cars:  # for aagregating constraints
            cars = cars.split(";")
        else:
            cars = [cars]
        for car in cars:
            if car not in n_cars:
                errors.append(car)

    if errors:
        logger.error(f"{errors} are not defined in network.")
        return False
    else:
        return True

def create_directory(d: str | Path, del_existing: bool = True) -> None:
    """Removes exising data and creates empty directory"""

    if isinstance(d, str):
        d = Path(d)

    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        return

    existing_files = [x.name for x in d.iterdir()]

    if not del_existing and existing_files:
        raise ValueError(f"Delete files {existing_files} in {str(d)} and rerun")
    elif del_existing and existing_files:
        for f in d.iterdir():
            if f.is_file():
                f.unlink()

    return


def get_sample_data(
    params: pd.DataFrame, sample: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Gets data structure for applying data"""

    data = {}
    p = params.set_index("name")
    for name in p.index:
        data[name] = {}
        data[name]["component"] = p.at[name, "component"]
        data[name]["carrier"] = p.at[name, "carrier"]
        data[name]["attribute"] = p.at[name, "attribute"]
        data[name]["range"] = p.at[name, "range"]  # absolute | percent
        data[name]["value"] = sample[name].to_dict()  # run: value
    return data

def get_set_value_data(
    set_values: pd.DataFrame, num_runs: int
) -> dict[str, dict[str, Any]]:
    """Gets data structure for applying data from set_values"""

    data = {}
    sv = set_values.set_index("name")
    for name in sv.index:
        data[name] = {}
        data[name]["component"] = sv.at[name, "component"]
        data[name]["carrier"] = sv.at[name, "carrier"]
        data[name]["attribute"] = sv.at[name, "attribute"]
        data[name]["range"] = sv.at[name, "range"]  # absolute | percent
        data[name]["value"] = {
            x: sv.at[name, "value"] for x in range(num_runs)
        }  # run: value
    return data


def _is_valid_ref_value(value: Any, car: str, attr: str) -> bool:
    """Checks if the ref value extracted is valid."""
    # edge case used to discout
    if (attr == "capital_cost") and car in ("oil", "waste"):
        return True
    elif (attr == "efficiency2") and (car == "gas production"):
        return True
    # main condition to satisty
    # if value == np.nan or value == 0:
    if value == np.nan:
        logger.info(f"Invalid reference value of {value} for {attr} {car}")
        return False
    else:
        return True


def calc_difference(ref: float, sampled: float, car: str, attr: str) -> float:
    """Calucates percent difference between original and sampled values."""

    assert _is_valid_ref_value(ref, car, attr)

    if (ref == 0) or (ref == np.nan):
        return np.nan
    else:
        return abs(sampled - ref) / ref * 100

def _apply_demand_response_marginal_cost(n: pypsa.Network, value: int | float) -> None:
    """Applied demand response to both forward and backwards directions."""

    df = getattr(n, "stores")

    assert all(df[df.carrier == "demand_response"].marginal_cost_storage != 0)

    fwd_slicer = df[
        (df.carrier == "demand_response") & (df.marginal_cost_storage < 0)
    ].index
    bck_slicer = df[
        (df.carrier == "demand_response") & (df.marginal_cost_storage > 0)
    ].index

    # forward demand reponse will have negative marginal cost

    getattr(n, "stores").loc[fwd_slicer, "marginal_cost_storage"] = value * (-1)

    # backwards demand response will have positive marginal cost

    getattr(n, "stores").loc[bck_slicer, "marginal_cost_storage"] = value


def _apply_static_sample(
    n: pypsa.Network, c: str, car: str, attr: str, value: int | float, absolute: bool
) -> dict[str, float]:
    """Applies a time independent value to the network.

    Returns value applied and difference from ref value.
    """
    df = getattr(n, c)
    slicer = df[df.carrier == car].index
    if absolute:
        # get metadata
        sampled = value
        ref = getattr(n, c).loc[slicer, attr].mean()
        diff = calc_difference(ref, sampled, car, attr)
        # apply value
        if car == "demand_response":
            _apply_demand_response_marginal_cost(n, value)
        else:
            getattr(n, c).loc[slicer, attr] = value
    else:
        # get metadata
        ref = getattr(n, c).loc[slicer, attr]
        if attr == "p_nom":  # only include existing capacity
            ref = ref[ref > 0]
            if ref.empty:
                logger.debug(f"No exsiting p_nom for {car}")
                ref = 1 # temporary correction to avoid divide by zero 
            else:
                ref = ref.mean()
        else:
            ref = ref.mean()
        multiplier = value / 100  # can be positive or negative
        sampled = ref + ref * multiplier
        diff = calc_difference(ref, sampled, car, attr)
        # apply value
        ref_slice = getattr(n, c).loc[slicer, attr]
        getattr(n, c).loc[slicer, attr] = ref_slice + ref_slice.mul(multiplier)

    return {
        "ref": round(ref, 5),  # original value applied to network
        "scaled": round(sampled, 5),  # new value applied to network
        "difference": round(diff, 5),  # percent diff between original and new
    }

def _apply_dynamic_sample(
    n: pypsa.Network, c: str, car: str, attr: str, value: int | float
) -> tuple[float, float]:
    """Applies a time dependent value to the network.

    Returns mean value applied and difference from mean ref value.
    """
    # apply value
    df_t = getattr(n, c)[attr]
    df_static = getattr(n, c.split("_t")[0])

    name_to_carrier = df_static.carrier.to_dict()
    name_carrier_map = {x: name_to_carrier[x] for x in df_t.columns}
    names = [x for x, y in name_carrier_map.items() if y == car]

    ref = getattr(n, c)[attr].loc[:, names]  # same as df_t.loc[...]
    multiplier = value / 100  # can be positive or negative
    getattr(n, c)[attr].loc[:, names] = ref + ref.mul(multiplier)

    # get metadata
    scaled = (ref + ref.mul(multiplier)).mean().mean()
    ref = ref.mean().mean()
    diff = calc_difference(ref, scaled, car, attr)

    assert scaled is not np.nan, f"Scaled {car} {attr} is np.nan"

    return {
        "ref": round(ref, 5),  # original mean value applied to network
        "scaled": round(scaled, 5),  # new mean value applied to network
        "difference": round(diff, 5),  # percent diff between original and new
    }


def _apply_cached_capex(
    n: pypsa.Network, car: str, data: dict[str, Any]
) -> dict[str, float]:
    try:
        cache = CapitalCostCache(**data)
        capex = cache.calculate_capex()
    except ValueError as ex:
        logger.error(f"Capital cost error with {car}")
        raise ValueError(ex)
    if car.endswith("battery_storage"):  # extra carrier
        pass
    sampled = _apply_static_sample(
        n, cache.component, car, "capital_cost", capex, "absolute"
    )
    return sampled


def _apply_cached_ch4_leakage(
    n: pypsa.Network, data: dict[str, Any], upstream: bool
) -> dict[str, float]:
    try:
        cache = MethaneLeakageCache(**data)
        leakage = cache.calculate_leakage()
    except ValueError as ex:
        logger.error("Methane Leakage error")
        raise ValueError(ex)

    if upstream:  # applies sampled value to gas production of nat gas
        sampled = _apply_static_sample(
            n, cache.component, "gas production", "efficiency3", leakage, "absolute"
        )
    else:  # applies same sampled value to all end-users of nat gas
        gas_buses = n.buses[n.buses.carrier == "gas"]
        carriers = (
            n.links[
                (n.links.bus0.isin(gas_buses.index))
                & ~(n.links.carrier.isin(["gas storage", "gas trade"]))
            ]
            .carrier.unique()
            .tolist()
        )
        for car in carriers:
            # sampled value will always be the same
            sampled = _apply_static_sample(
                n, cache.component, car, "efficiency3", leakage, "absolute"
            )

    return sampled


def _cache_attr(
    cache: dict[str, float], c: str, car: str, attr: str, value: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Caches attribute.

    Returns updated cache and sample metadata
    """
    if car not in cache:
        cache[car] = {"component": c}
    cache[car][attr] = value
    sampled = {
        "ref": np.nan,
        "scaled": value,  # already absolute
        "difference": np.nan,
    }
    return cache, sampled


def _get_rps_value(n: pypsa.Network, rps: pd.DataFrame) -> float:
    """Gets mean RPS requirement in MWh. Used only for SEE sample extraction."""

    if rps.empty:
        return
    
    demand = []
    
    portfolio_standards = concat_rps_standards(n, rps)

    # Iterate through constraints
    for _, constraint_row in portfolio_standards.iterrows():
        
        region_buses, region_gens = get_rps_eligible(n, constraint_row.region, constraint_row.carrier)
        
        if region_buses.empty:
            continue

        if not region_gens.empty:
            region_demand = get_rps_demand_gsa(n, constraint_row.planning_horizon, region_buses)
            region_demand *= 1.1 # approximate for transportation elec demenad
            demand.append(constraint_row.pct * region_demand)
            
    return sum(demand) / len(demand)

def _get_ng_trade(n: pypsa.Network, trade: pd.DataFrame, direction: str) -> float:
    """Gets RHS import/export limit."""
    data = format_raw_ng_trade_data(trade, " trade")
    links = get_ng_trade_links(n, direction)
    links_in_scope = [x for x in links if x in data.index]
    return data.loc[links_in_scope, "rhs"].sum()


def _get_gshp_multiplier(n: pypsa.Network, pop: pd.DataFrame) -> float:
    """Gets fractional multipler to apply to GSHP capacity."""

    gshp = n.links[n.links.index.str.endswith("gshp")].copy()
    fraction = get_urban_rural_fraction(pop)
    gshp["urban_rural_fraction"] = gshp.bus0.map(fraction)
    return gshp.urban_rural_fraction.mean() / 100

def _get_ev_generation_limit(
    n: pypsa.Network, mode: str, policy: pd.DataFrame
) -> float:
    """Limit on yearly generation from EVs"""
    mode_mapper = {
        "light_duty": "lgt",
        "med_duty": "med",
        "heavy_duty": "hvy",
        "bus": "bus",
    }

    assert len(n.investment_periods) == 1
    investment_period = n.investment_periods[0]

    dem_names = n.loads[n.loads.carrier == f"trn-veh-{mode_mapper[mode]}"].index
    dem = n.loads_t["p_set"][dem_names]

    evs = n.links[n.links.carrier == f"trn-elec-veh-{mode_mapper[mode]}"].index
    ratio = policy.at[investment_period, mode] / 100  # input is percentage
    eff = n.links.loc[evs].efficiency.mean()

    return dem.loc[investment_period].sum().sum() * ratio / eff

def _get_constraint_sample(
    n: pypsa.Network, c: str, car: str, attr: str, sample: float, **kwargs
) -> tuple[dict[str, float], dict[str, float]]:
    """Gets sample data to apply to the RHS of the constraint.

    Returns meta data specific to constraints. This is a subset of all metadata.
    """
    if attr == "lv":
        ref = get_existing_lv(n)
        assert sample <= 1
        scaled = ref * (1 + sample)
    elif attr == "rps":
        rps = kwargs.get("rps", pd.DataFrame())
        if rps.empty:
            raise ValueError("No ref RPS provided.")
        ref = _get_rps_value(n, rps)
        scaled = ref * sample
    elif attr == "ces":
        ces = kwargs.get("ces", pd.DataFrame())
        if ces.empty:
            raise ValueError("No ref CES provided.")
        ref = _get_rps_value(n, ces)
        scaled = ref * sample
    elif attr == "tct": # already absolute
        ref = sample
        scaled = sample
    elif attr == "co2L": # already absolute
        ref = sample
        scaled = sample
    elif attr == "nat_gas_import":
        # domestic trade
        ng_domestic = kwargs.get("ng_domestic", pd.DataFrame())
        if ng_domestic.empty:
            raise ValueError("No domestic NG trade data provided.")
        dom = _get_ng_trade(n, ng_domestic, "imports")
        # international trade
        ng_international = kwargs.get("ng_international", pd.DataFrame())
        if ng_international.empty:
            raise ValueError("No international NG trade data provided.")
        itl = _get_ng_trade(n, ng_domestic, "imports")
        # return total imports
        ref = dom + itl
        scaled = ref + (ref * sample)
    elif attr == "nat_gas_export":
        # domestic trade
        ng_domestic = kwargs.get("ng_domestic", pd.DataFrame())
        if ng_domestic.empty:
            raise ValueError("No domestic NG trade data provided.")
        dom = _get_ng_trade(n, ng_domestic, "imports")
        # international trade
        ng_international = kwargs.get("ng_international", pd.DataFrame())
        if ng_international.empty:
            raise ValueError("No international NG trade data provided.")
        itl = _get_ng_trade(n, ng_domestic, "exports")
        # return total exports
        ref = dom + itl
        scaled = ref * sample
    elif attr == "ev_policy":
        ev_policy = kwargs.get("ev_policy", pd.DataFrame())
        if ev_policy.empty:
            raise ValueError("No EV policy data provided.")
        car_2_mode = {
            "trn-elec-veh-lgt": "light_duty",
            "trn-elec-veh-med": "med_duty",
            "trn-elec-veh-hvy": "heavy_duty",
            "trn-elec-veh-bus": "bus",
        }
        ref = _get_ev_generation_limit(n, car_2_mode[car], ev_policy)
        scaled = ref + (ref * sample)
    elif attr == "gshp":
        pop = kwargs.get("population", pd.DataFrame())
        if pop.empty:
            raise ValueError("No population data provided.")
        ref = _get_gshp_multiplier(n, pop)
        scaled = ref * sample
    else:
        raise ValueError(f"Bad control flow for get_constraint_sample: {attr}")


    meta = {
        "component": c,
        "carrier": car,
        "attribute": attr,
        "value": sample, # actual solve network still ingests unscaled value
    }
    sampled = {
        "ref": ref,
        "scaled": scaled,
        "difference": round(abs(scaled - ref) / ref * 100, 5),
    }

    return meta, sampled


def apply_sample(
    n: pypsa.Network, sample: dict[str, dict[str, Any]], run: int, **kwargs
) -> tuple[list[float], dict[dict[str, str | float]], pd.DataFrame]:
    """Applies a sample to a network for a single model run.

    As there are some intermediate calculations, some data is cached and applied at the end.

    This will modify the network! Pass a copy of a network if you dont want to modify the
    reference network.

    Returns:
        list[float]:
            scaled sample for this model run
        dict[dict[str, str | float]]:
            metadata associated with model run
        pd.DataFrame:
            constraint samples to pass into (unscaled)
    """

    meta = {}
    cached = {}  # for internediate calculations

    # save constraint metadata in smaller file as it needs to be read in with every model
    # this is the same data as in the bigger meta file tho.
    meta_constraints = {}

    # return values to scale the EE
    #############################
    ## ORDER MUST BE PRESERVED ##
    #############################
    scaled_sample = []
    #############################
    ## ORDER MUST BE PRESERVED ##
    #############################

    for name, data in sample.items():
        c = data["component"]
        car = data["carrier"]
        attr = data["attribute"]
        absolute = True if data["range"] == "absolute" else False
        value = round(data["value"][run], 5)

        # if the applied value is an intermediate calculation
        if attr in CACHED_ATTRS and absolute:
            cached, sampled = _cache_attr(cached, c, car, attr, value)
        # if the value is applied to the RHS of the constraint
        # note that the constraints arnt actually applied here
        elif attr in CONSTRAINT_ATTRS:
            mc, sampled = _get_constraint_sample(n, c, car, attr, value, **kwargs)
            meta_constraints[str(name)] = mc
        # if the value is applied to a time-dependent value
        elif c.endswith("_t"):
            if absolute:
                raise ValueError(f"{attr} for {car} can not be absolute")
            if car == "res-elec" and attr == "p_set":
                pass
            sampled = _apply_dynamic_sample(n, c, car, attr, value)
        # if the value is applied to a time-independent value
        else:
            sampled = _apply_static_sample(n, c, car, attr, value, absolute)

        #####################
        # PRESERVING ORDER ##
        #####################
        scaled_sample.append(sampled["scaled"])
        #####################
        # PRESERVING ORDER ##
        #####################

        original = sampled["ref"]
        ss = sampled["scaled"]
        diff = sampled["difference"]

        meta[str(name)] = {
            "component": c,
            "carrier": car,
            "attribute": attr,
            "range": data["range"],
            # value passed in from the unscaled sample
            "sample": value,
            # original network value
            "original": "" if original == np.nan else float(original),
            # difference between applied value and original network value
            "diff": "" if diff == np.nan else float(diff),
            # scaled sample value
            "scaled_sample": "" if ss == np.nan else float(ss),
        }

    # as these values are intermediate, they are not part of the actual sample
    for car, data in cached.items():
        if car == "leakage_upstream":
            sampled = _apply_cached_ch4_leakage(n=n, data=data, upstream=True)
            attr = "efficiency3"
        elif car == "leakage_downstream":
            sampled = _apply_cached_ch4_leakage(n=n, data=data, upstream=False)
            attr = "efficiency3"
        else:
            sampled = _apply_cached_capex(n, car, data)
            attr = "capital_cost"

        meta[str(car)] = {
            "component": data["component"],
            "carrier": car,
            "attribute": attr,
            "range": "absolute",  # requirement for cached data
            "sample": float(sampled["scaled"]),  # since absolute, scaled = sample
            "original": float(sampled["ref"]),
            "diff": float(sampled["difference"]),
            "scaled_sample": float(sampled["scaled"]),
        }

    if not meta_constraints:
        meta_constraints = pd.DataFrame(
            columns=["name", "component", "carrier", "attribute", "value"]
        )
    else:
        meta_constraints = pd.DataFrame.from_dict(
            meta_constraints, orient="index"
        ).reset_index(names="name")

    return scaled_sample, meta, meta_constraints

def apply_load_shedding(n: pypsa.Network) -> None:
    """Intersect between macroeconomic and surveybased willingness to pay

    - (Fig 2) http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
    - (pg 21) https://ssrn.com/abstract=3951809
    - (Table ES1) https://www.osti.gov/servlets/purl/1172643
    """

    n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")

    # Table ES1 - average of 1hr Cost per Unserved kWh over all customers
    shed_cost = 106000

    ###
    # This applies at power sector level
    ###

    buses_i = n.buses.query("carrier == 'AC'").index

    n.madd(
        "Generator",
        buses_i,
        " load",
        bus=buses_i,
        carrier="load",
        marginal_cost=shed_cost,
        p_nom=0,  # kW
        capital_cost=0,
        p_nom_extendable=True,
    )

    ###
    # This applies at LPG transport level
    ###

    buses1 = n.links[
        (n.links.carrier.str.startswith("trn-"))
        & (n.links.carrier.str.endswith("-veh"))
        & (n.links.carrier.str.contains("-lpg-"))
    ].bus1.to_list()

    buses2 = n.links[
        (n.links.carrier.str.startswith("trn-"))
        & (n.links.carrier.str.endswith(("-air", "-rail", "-boat")))
    ].bus1.to_list()

    buses_i = buses1 + buses2

    n.madd(
        "Generator",
        buses_i,
        " load",
        bus=buses_i,
        carrier="load",
        marginal_cost=shed_cost,
        p_nom=0,
        capital_cost=0,
        p_nom_extendable=True,
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        param_file = snakemake.input.parameters
        sample_file = snakemake.input.sample_file
        set_values_file = snakemake.input.set_values_file
        base_network_file = snakemake.input.network
        root_dir = Path(snakemake.params.root_dir)
        meta_yaml = snakemake.params.meta_yaml
        meta_csv = snakemake.params.meta_csv
        scaled_sample_file = snakemake.output.scaled_sample
        pop_f = snakemake.input.pop_layout_f
        ng_dommestic_f = snakemake.input.ng_domestic_f
        ng_international_f = snakemake.input.ng_international_f
        rps_f = snakemake.input.rps_f
        ces_f = snakemake.input.ces_f
        ev_policy_f = snakemake.input.ev_policy_f
        configure_logging(snakemake)
    else:
        param_file = "results/caiso/gsa/parameters.csv"
        sample_file = "results/caiso/gsa/sample.csv"
        set_values_file = ""
        base_network_file = "results/caiso/base.nc"
        root_dir = Path("results/caiso/gsa/modelruns/")
        meta_yaml = True
        meta_csv = True
        scaled_sample_file = "results/caiso/gsa/scaled_sample.csv"
        pop_f = "results/caiso/constraints/pop_layout.csv"
        ng_dommestic_f = "results/caiso/constraints/ng_domestic.csv"
        ng_international_f = "results/caiso/constraints/ng_international.csv"
        rps_f = "results/caiso/constraints/rps.csv"
        ces_f = "results/caiso/constraints/ces.csv"
        ev_policy_f = "results/caiso/constraints/ev_policy.csv"

    params = pd.read_csv(param_file)
    sample = pd.read_csv(sample_file)
    base_n = pypsa.Network(base_network_file)

    apply_load_shedding(base_n)

    # check carrier here as it requires reading in network
    assert is_valid_carrier(base_n, params)

    sample_data = get_sample_data(params, sample)

    # add in pre-defined valies to the sample.
    if set_values_file:
        set_values = pd.read_csv(set_values_file)
        num_runs = len(sample)
        set_values_data = get_set_value_data(set_values, num_runs)
        assert all([x not in set_values_data for x in sample_data])
        sample_data = sample_data | set_values_data
        scaled_scample_columns = sample.columns.to_list() + set_values.name.to_list()
    else:
        scaled_scample_columns = sample.columns

    scaled_sample = []

    # for scaling constraint data, we need external data sources
    constraint_data = {
        "rps": pd.read_csv(rps_f),
        "ces": pd.read_csv(ces_f),
        "ng_domestic": pd.read_csv(ng_dommestic_f, index_col=0),
        "ng_international": pd.read_csv(ng_international_f, index_col=0),
        "population": pd.read_csv(pop_f),
        "ev_policy": pd.read_csv(ev_policy_f, index_col=0),
    }

    # MUST BE 'sample' and NOT 'sample_data' as set_values is added to the 'sample_data'

    for run in range(len(sample)):
        n = base_n.copy()

        scaled, meta, meta_constraints = apply_sample(
            n, sample_data, run, **constraint_data
        )

        scaled_sample.append(scaled)

        n_save_name = Path(root_dir, str(run), "n.nc")
        meta_constraints_save_name = Path(root_dir, str(run), "constraints.csv")

        n.export_to_netcdf(n_save_name)

        meta_constraints.to_csv(meta_constraints_save_name, index=False)

        if meta_yaml:
            meta_save_name = Path(root_dir, str(run), "meta.yaml")
            with open(meta_save_name, "w") as f:
                yaml.dump(meta, f)

        if meta_csv:
            meta_save_name = Path(root_dir, str(run), "meta.csv")
            meta_df = pd.DataFrame.from_dict(meta).T
            meta_df.to_csv(meta_save_name, index=True)

    ss = pd.DataFrame(scaled_sample, columns=scaled_scample_columns).round(5)
    ss.to_csv(scaled_sample_file, index=False)
