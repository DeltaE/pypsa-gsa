"""Generates Total Capacity Tartget (TCT) data based on AEO projections."""

import math
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
    # assumed higher then ref for lower bounds cause for high demand and high rps
    # runs, infeasabilities can occur due to REC trading not being captured
    # to models outside the scope
    "battery": {
        "ref_growth": 1725,  # ref * 2 = 862.5 * 2 = 1725
        "max_growth": 2155,
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
    "battery": ["4hr_battery_storage", "8hr_battery_storage", "battery"],
}

COMPONENTS = {
    "biomass": "generator",
    "geothermal": "generator",
    "steam": "generator",
    "nuclear": "generator",
    "wind": "generator",
    "solar": "generator",
    "hydro": "generator",
    "ccgt": "link",
    "ocgt": "link",
    "coal": "link",
    "battery": "storage_units",
}

TCT_COLUMNS = ["name", "planning_horizon", "region", "carrier", "min", "max"]


def _get_current_capactity(n: pypsa.Network, cars: list[str], component: str) -> float:
    if component.startswith("generator"):
        value = n.generators[n.generators.carrier.isin(cars)].p_nom.sum()
    elif component.startswith("link"):
        value = n.links[n.links.carrier.isin(cars)].p_nom.sum()
    elif component.startswith("storage_unit"):
        value = n.storage_units[n.storage_units.carrier.isin(cars)].p_nom.sum()
    else:
        raise ValueError(f"Invalid component: {component}")
    return round(value, 5)


def adjust_zero_capacity(n: pypsa.Network, cars: list[str], component: str) -> float:
    """Allows for some growth if current capacity is zero."""

    total_cap_gen = _get_current_capactity(
        n, [x for x, y in COMPONENTS.items() if y == "generator"], "generator"
    )
    total_cap_link = _get_current_capactity(
        n, [x for x, y in COMPONENTS.items() if y == "link"], "link"
    )
    total_cap = total_cap_gen + total_cap_link

    current_cap = _get_current_capactity(n, cars, COMPONENTS[component])

    if current_cap < 1 and any(x in cars for x in ["solar", "onwind"]):
        current_cap = total_cap * 0.10
    elif current_cap < 1 and any(x in cars for x in ["CCGT", "OCGT"]):
        current_cap = total_cap * 0.05

    return current_cap


def get_tct_data(n: pypsa.Network, ccs_limit: float | None = None) -> pd.DataFrame:
    """Gets TCT constraint data."""

    if not ccs_limit:
        ccs_limit = 0

    planning_year = n.investment_periods[0]

    data = []

    for name, cars in CARRIERS.items():
        cap = _get_current_capactity(n, cars, COMPONENTS[name])

        if cap < 1:
            cap = adjust_zero_capacity(n, cars, name)

        ref_growth = GROWTHS[name]["ref_growth"]
        if ref_growth < 100:
            ref_cap = cap + 0.1
        else:
            ref_cap = float(math.ceil(cap * ref_growth / 100))

        if ref_cap < 0.1:
            ref_cap = 0.1

        tct = [f"tct_{name}", planning_year, "all", ",".join(cars), "", ref_cap]
        data.append(tct)

        if name == "ccgt":
            ccs_cap = float(math.ceil(ref_cap * ccs_limit / 100))
            tct = [f"tct_{name}_ccs", planning_year, "all", "CCGT-95CCS", "", ccs_cap]
            data.append(tct)

    return pd.DataFrame(data, columns=TCT_COLUMNS)


def get_gsa_tct_data(n: pypsa.Network, ccs_limit: float | None = None) -> pd.DataFrame:
    """Gets formatted TCT data to pass into GSA."""

    if not ccs_limit:
        ccs_limit = 0

    data = []

    for name, cars in CARRIERS.items():
        cap = _get_current_capactity(n, cars, COMPONENTS[name])

        # allow some growth for new technologies
        if cap < 1:
            cap = adjust_zero_capacity(n, cars, name)

        ref_growth = GROWTHS[name]["ref_growth"]
        if ref_growth < 100:
            min_value = cap
            max_value = cap * GROWTHS[name]["max_growth"] / 100
        else:
            min_value = cap * GROWTHS[name]["ref_growth"] / 100
            max_value = cap * GROWTHS[name]["max_growth"] / 100

        min_value = round(min_value, 1)
        max_value = round(max_value, 1)

        if max_value < 0.1:
            max_value = 0.1

        if abs(min_value - max_value) < 0.0001:
            logger.info(f"No limits created for {name}")
            continue

        gsa = [
            f"tct_{name}",
            f"tct_{name}",
            f"{name.capitalize()} Max Capacity",
            COMPONENTS[name],
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

        if name == "ccgt":
            min_value = float(math.ceil(min_value * ccs_limit / 100))
            max_value = float(math.ceil(max_value * ccs_limit / 100))

            if abs(min_value - max_value) < 0.0001:
                min_value -= 0.1
                max_value += 0.1
                if min_value < 0:
                    min_value = 0

            gsa = [
                f"tct_{name}_ccs",
                f"tct_{name}_ccs",
                f"{name.capitalize()}-95CCS Max Capacity",
                COMPONENTS[name],
                "CCGT-95CCS",
                "tct",
                "absolute",
                "per_unit",
                min_value,
                max_value,
                "Assumed in config file",
                "",
            ]
            data.append(gsa)

    return pd.DataFrame(
        data,
        columns=GSA_COLUMNS,
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        network = snakemake.input.network
        include_tct = snakemake.params.include_tct
        tct_aeo_f = snakemake.output.tct_aeo
        tct_gsa_f = snakemake.output.tct_gsa
        ccs_limit = snakemake.params.ccs_limit  # as a percentage of max ccgt cap
    else:
        network = "config/pypsa-usa/nd/elec_s20_c2m_ec_lv1.0_3h_E-G.nc"
        include_tct = True
        tct_aeo_f = "results/testing/generated/tct_aeo.csv"
        tct_gsa_f = "results/testing/generated/tct_aeo.csv"
        ccs_limit = 50  # as a percentage of max ccgt cap

    n = pypsa.Network(network)
    assert len(n.investment_periods) == 1

    if include_tct:
        assert tct_aeo_f and tct_gsa_f, (
            "tct_aeo_f and tct_gsa_f must be provided for TCT"
        )

        tct_aeo = get_tct_data(n, ccs_limit)
        tct_gsa = get_gsa_tct_data(n, ccs_limit)

    else:
        logger.info("No TCT constraint data generated")
        tct_aeo = pd.DataFrame(columns=TCT_COLUMNS)
        tct_gsa = pd.DataFrame(columns=GSA_COLUMNS)

    tct_aeo.to_csv(tct_aeo_f, index=False)
    tct_gsa.to_csv(tct_gsa_f, index=False)
