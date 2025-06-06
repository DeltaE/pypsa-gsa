"""Independent script to collect result data into a database."""

import pandas as pd
from pathlib import Path
import json

from components.utils import ISOS

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

root = Path(__file__).parent.parent

ROUND_TO = 5

PARAM_ATTRIBUTE_NICE_NAMES = {
    "co2L": "Emission Limit",
    "discount_rate": "Discount Rate",
    "e_min_pu": "Minimum Storage Level",
    "efficiency": "Efficiency",
    "efficiency_store": "Efficiency",
    "ev_policy": "EV Policy",
    "fixed_cost": "Fixed Cost",
    "gshp": "Ground Source Heat Pump",
    "gwp": "Gloabl Warming Potential",
    "itc": "Investment Tax Credit",
    "leakage": "Methane Leakage",
    "lifetime": "Lifetime",
    "lv": "Transmission Expansion Volume",
    "marginal_cost": "Marginal Cost",
    "marginal_cost_storage": "Marginal Cost",
    "nat_gas_export": "Natural Gas Trade",
    "nat_gas_import": "Natural Gas Trade",
    "occ": "Overnight Capital Cost",
    "p_max_pu": "Capacity Factor",
    "p_nom": "Existing Capacity",
    "p_set": "Loads",
    "tct": "Capacity Limits",
    "vmt_per_year": "Lifetime",
}

PWR_CARRIERS = [
    "CCGT",
    "CCGT-95CCS",
    "OCGT",
    "biomass",
    "coal",
    "geothermal",
    "hydro",
    "lpg",
    "nuclear",
    "offwind_floating",
    "oil",
    "onwind",
    "solar",
    "waste",
]


def _round_df(df: pd.DataFrame) -> pd.DataFrame:
    """Converts small values to zero."""
    return df.mask(df.abs() < 1e-8, 0).round(ROUND_TO)


def get_param_names(root: Path, iso: str, mode: str) -> dict[str, str]:
    """Gets names and nice_names for the parameter data."""

    assert mode in ("gsa", "ua"), f"Invalid mode: {mode}. Must be 'gsa' or 'ua'."

    data_f = Path(root, "results", iso, mode, "parameters.csv")
    df = pd.read_csv(data_f, index_col=0)[["group", "nice_name"]]
    logger.debug(f"Collecting nice names from {data_f}.")
    df = df.drop_duplicates()
    return df.set_index("group")["nice_name"].to_dict()


def get_result_names(root: Path, iso: str, mode: str) -> dict[str, str]:
    """Gets names and nice_names for the result data."""

    assert mode in ("gsa", "ua"), f"Invalid mode: {mode}. Must be 'gsa' or 'ua'."

    data_f = Path(root, "results", iso, mode, "results.csv")
    df = pd.read_csv(data_f, index_col=0)
    logger.debug(f"Collecting nice names from {data_f}.")
    return df.nice_name.to_dict()


def collect_sa(root: Path, iso: str, results: list[str]) -> pd.DataFrame:
    """Collects the SA data from the results directory."""

    data_dir = Path(root, "results", iso, "gsa", "SA")
    logger.debug(f"Collecting SA data for {iso} from {data_dir}.")

    inputs = [Path(data_dir, f"{x}.csv") for x in results]

    dfs = []
    for f in inputs:
        name = f.stem
        df = pd.read_csv(f, index_col=0).mu_star
        df.name = name
        dfs.append(df)
    df = _round_df(pd.concat(dfs, axis=1))
    df["iso"] = iso
    df = df.reset_index(names=["param"])
    return df.set_index(["param", "iso"])


def collect_runs(root: Path, iso: str, mode: str, results: list[str]) -> pd.DataFrame:
    """Collects the model run results data from the results directory."""

    assert mode in ("gsa", "ua"), f"Invalid mode: {mode}. Must be 'gsa' or 'ua'."

    data_dir = Path(root, "results", iso, mode, "results")
    logger.debug(f"Collecting model run results data for {iso} from {data_dir}.")

    inputs = [Path(data_dir, f"{x}.csv") for x in results]

    dfs = []
    for f in inputs:
        name = f.stem
        try:
            df = pd.read_csv(f, index_col=1)
        except FileNotFoundError:
            logger.warning(f"No file for for {f}.")
            continue
        df = df.rename(columns={"value": name})
        dfs.append(df)
    df = _round_df(pd.concat(dfs, axis=1))
    df["iso"] = iso
    df = df.reset_index(names=["run"])
    return df.set_index(["run", "iso"])


def get_empty_sa() -> pd.DataFrame:
    """Returns an empty dataframe for the SA data."""
    return pd.DataFrame(columns=["param", "iso"]).set_index(["param", "iso"])


def get_empty_run() -> pd.DataFrame:
    """Returns an empty dataframe for the run data."""
    return pd.DataFrame(columns=["param", "iso"]).set_index(["param", "iso"])


def get_base_params_file(root: Path, iso: str) -> pd.DataFrame:
    """Returns the base parameters file for shared between all ISOs."""
    df = pd.read_csv(Path(root, "results", iso, "gsa", "parameters.csv"))
    # remove iso specific data
    df = df[~((df.name.str.startswith("tct_")) | (df.name == ("emission_limit")))]
    df["iso"] = "all"
    return df


def get_iso_params(root: Path, iso: str) -> pd.DataFrame:
    """Returns the parameters file for the given ISO and mode."""
    df = pd.read_csv(Path(root, "results", iso, "gsa", "parameters.csv"))
    # track iso specific data
    df = df[(df.name.str.startswith("tct_")) | (df.name == ("emission_limit"))]
    df["iso"] = iso
    return df


def assign_parameter_filters(params: pd.DataFrame) -> pd.DataFrame:
    """Assigns nice names to the parameter attributes."""

    def _assign_sector(carrier: str) -> str:
        prefix = carrier.split("-")[0]
        if (prefix == "com") | (prefix == "res"):
            return "Service"
        elif prefix == "ind":
            return "Industry"
        elif prefix == "trn":
            return "Transport"
        elif carrier in PWR_CARRIERS:
            return "Power"
        elif carrier.split(";")[0] in PWR_CARRIERS:
            return "Power"
        elif carrier.endswith("_battery_storage"):
            return "Power"
        elif carrier in ["demand_response", "load"]:
            return "Demand Response"
        elif carrier.startswith("gas "):
            return "Natural Gas"
        elif carrier == "AC":
            return "Transmission"
        elif carrier == "co2":
            return "Carbon"
        else:
            raise ValueError(f"Invalid carrier: {carrier}")

    params["attribute_nice_name"] = params.attribute.map(PARAM_ATTRIBUTE_NICE_NAMES)
    params["sector"] = params.carrier.map(_assign_sector)
    fuel_cost_mask = params.component == "stores_t"
    params.loc[fuel_cost_mask, "sector"] = "Primary Fuel"
    return params


if __name__ == "__main__":
    # sensitivity measures

    dfs = []
    for iso in ISOS:
        iso_data = Path(root, "results", iso, "gsa", "SA")
        if not iso_data.exists():
            logger.warning(f"No gsa sa data for '{iso}'")
            dfs.append(get_empty_sa())
        else:
            sa_names = get_result_names(root, iso, "gsa")
            df = collect_sa(root, iso, sa_names.keys())
            dfs.append(df)

    if not dfs:
        logger.error("No data found.")
        raise ValueError("No ISO data found.")

    sa = pd.concat(dfs, axis=0)
    sa.to_csv(Path(root, "dashboard", "data", "sa.csv"), index=True)

    # model run results ua

    dfs = []
    for iso in ISOS:
        iso_data = Path(root, "results", iso, "ua", "results")
        if not iso_data.exists():
            logger.warning(f"No ua model run data for '{iso}'")
            dfs.append(get_empty_run())
        else:
            result_names = get_result_names(root, iso, "ua")
            df = collect_runs(root, iso, "ua", result_names.keys())
            dfs.append(df)
    if not dfs:
        logger.error("No data found.")
        raise ValueError("No ISO data found.")

    sa = pd.concat(dfs, axis=0)
    sa.to_csv(Path(root, "dashboard", "data", "ua_runs.csv"), index=True)

    # model parameters

    dfs = []
    base_params = pd.DataFrame()
    for iso in ISOS:
        iso_data = Path(root, "results", iso, "gsa")
        if not iso_data.exists():
            logger.warning(f"No parameter data for '{iso}'")
        else:
            if base_params.empty:
                base_params = get_base_params_file(root, iso)
                dfs.append(base_params)
            iso_params = get_iso_params(root, iso)
            dfs.append(iso_params)
    if not dfs:
        logger.error("No data found.")
        raise ValueError("No ISO data found.")

    params = pd.concat(dfs, axis=0)
    params = assign_parameter_filters(params)
    params = params[
        [
            "name",
            "group",
            "nice_name",
            "iso",
            "component",
            "carrier",
            "attribute",
            "attribute_nice_name",
            "sector",
            "range",
            "unit",
            "min_value",
            "max_value",
            "source",
            "notes",
        ]
    ]
    params.to_csv(Path(root, "dashboard", "data", "parameters.csv"), index=False)

    # get nice names

    for iso in ISOS:
        try:
            sa_params = get_param_names(root, iso, "gsa")
            sa_results = get_result_names(root, iso, "gsa")
        except FileNotFoundError:
            logger.debug(f"No gsa nice names for {iso}.")
            pass
        else:
            continue

    with open(Path(root, "dashboard", "data", "sa_params.json"), "w") as f:
        json.dump(sa_params, f, indent=4)

    with open(Path(root, "dashboard", "data", "sa_results.json"), "w") as f:
        json.dump(sa_results, f, indent=4)

    for iso in ISOS:
        try:
            ua_params = get_param_names(root, iso, "ua")
            ua_results = get_result_names(root, iso, "ua")
        except FileNotFoundError:
            logger.debug(f"No ua nice names for {iso}.")
            pass
        else:
            continue

    with open(Path(root, "dashboard", "data", "ua_params.json"), "w") as f:
        json.dump(ua_params, f, indent=4)

    with open(Path(root, "dashboard", "data", "ua_results.json"), "w") as f:
        json.dump(ua_results, f, indent=4)
