"""Generates Total Capacity Tartget (TCT) data based on AEO projections."""

import pandas as pd
import pypsa
from constants import GSA_COLUMNS

import logging
logger = logging.getLogger(__name__)

# user must populate this from the AEO notebook
# all values in percent
# ref_growth is the 2023 -> 2035 ref case growth
# max_growth is maximum growth over all scenarios by 2035
GROWTHS = {
    "biomass": {
        "ref_growth": 100,  # assumed
        "max_growth": 102,  # assumed
    },
    "geothermal": {
        "ref_growth": 100,  # assumed
        "max_growth": 102,  # assumed
    },
    "steam": {
        "ref_growth": 71.69,
        "max_growth": 100,
    },
    "nuclear": {
        "ref_growth": 90.18,
        "max_growth": 102.4,
    },
    "wind": {
        "ref_growth": 282.5,
        "max_growth": 346.5,
    },
    "solar": {
        "ref_growth": 282.5,
        "max_growth": 346.5,
    },
    "hydro": {
        "ref_growth": 100,  # assumed
        "max_growth": 102,  # assumed
    },
    "ccgt": {
        "ref_growth": 105.3,
        "max_growth": 117.8,
    },
    "ocgt": {
        "ref_growth": 164.8,
        "max_growth": 188.8,
    },
    "coal": {
        "ref_growth": 46.1,
        "max_growth": 100,
    },
}

CARRIERS = {
    "biomass": ["biomass"],
    "geothermal": ["geothermal"],
    "steam": ["waste", "oil"],
    "nuclear": ["nuclear"],
    "wind": ["onwind", "offwind_floating"],
    "solar": ["solar"],
    "hydro": ["hydro"],
    "ccgt": ["CCGT", "CCGT-95CCS"],
    "ocgt": ["OCGT"],
    "coal": ["coal"],
}

TCT_COLUMNS = ["name", "planning_horizon", "region", "carrier", "min", "max"]


def _get_current_capactity(n: pypsa.Network, cars: list[str]) -> float:
    gens = n.generators[n.generators.carrier.isin(cars)].p_nom.sum()
    links = n.links[n.links.carrier.isin(cars)].p_nom.sum()
    return round(gens + links, 5)


def get_tct_data(n: pypsa.Network) -> pd.DataFrame:
    """Gets TCT constraint data."""

    planning_year = n.investment_periods[0]

    data = []

    for name, cars in CARRIERS.items():
        cap = _get_current_capactity(n, cars)
        ref_growth = GROWTHS[name]["ref_growth"]
        if ref_growth < 100:
            ref_cap = cap + 0.1
        else:
            ref_cap = round(cap * ref_growth / 100, 5)

        tct = [f"tct_{name}", planning_year, "all", ",".join(cars), "", ref_cap]
        data.append(tct)

    return pd.DataFrame(data, columns=TCT_COLUMNS)


def get_gsa_tct_data(n: pypsa.Network) -> pd.DataFrame:
    """Gets formatted TCT data to pass into GSA."""

    data = []

    for name, cars in CARRIERS.items():
        cap = _get_current_capactity(n, cars)
        ref_growth = GROWTHS[name]["ref_growth"]
        if ref_growth < 100:
            min_value = cap
            max_value = cap * GROWTHS[name]["max_growth"] / 100
        else:
            min_value = cap * GROWTHS[name]["ref_growth"] / 100
            max_value = cap * GROWTHS[name]["max_growth"] / 100

        min_value = round(min_value / cap, 5)
        max_value = round(max_value / cap, 5)

        if abs(min_value - max_value) < 0.0001:
            logger.info(f"No limits created for {name}")
            continue

        gsa = [
            f"tct_{name}",
            f"tct_{name}",
            f"{name.capitalize()} Max Capacity",
            "generator",
            ";".join(cars),
            "tct",
            "absolute",
            "per_unit",
            min_value,
            max_value,
            "https://www.eia.gov/outlooks/aeo/",
            "See notebook in repository",
        ]
        data.append(gsa)

    return pd.DataFrame(
        data,
        columns=GSA_COLUMNS,
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        network = snakemake.params.network
        tct_aeo_f = snakemake.output.tct_aeo
        tct_gsa_f = snakemake.output.tct_gsa
    else:
        network = ""
        tct_aeo_f = ""
        tct_gsa_f = ""

    n = pypsa.Network(network)
    assert len(n.investment_periods) == 1
    tct_aeo = get_tct_data(n)
    tct_gsa = get_gsa_tct_data(n)

    tct_aeo.to_csv(tct_aeo_f, index=False)
    tct_gsa.to_csv(tct_gsa_f, index=False)
