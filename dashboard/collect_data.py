"""Independent script to collect result data into a database."""

from typing import Any
import pandas as pd
from pathlib import Path
import json

from components.utils import STATES

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

root = Path(__file__).parent.parent

ROUND_TO = 5

PARAM_ATTRIBUTE_NICE_NAMES = {
    "co2L": "Constraints",
    "discount_rate": "Discount Rate",
    "e_min_pu": "Natural Gas",  # only one e_min_pu
    "efficiency": "Efficiency",
    "efficiency_store": "Efficiency",
    "efficiency2": "Efficiency",
    "ev_policy": "Constraints",
    "fixed_cost": "Fixed Cost",
    "gshp": "Constraints",
    "gwp": "Methane",
    "itc": "Investment Tax Credit",
    "leakage": "Methane",
    "lifetime": "Lifetime",
    "lv": "Constraints",
    "marginal_cost": "Marginal Cost",
    "marginal_cost_storage": "Marginal Cost",
    "nat_gas_export": "Natural Gas",
    "nat_gas_import": "Natural Gas",
    "occ": "Overnight Capital Cost",
    "p_max_pu": "Capacity Factor",
    "p_nom": "Existing Capacity",
    "p_set": "Loads",
    "tct": "Capacity Limits",
    "vmt_per_year": "Lifetime",
    "rps": "Constraints",
    "ces": "Constraints",
    "elec_trade": "Constraints",
    "landuse": "Constraints",
    "rec": "Marginal Cost",
    "imports_co2": "Import Emissions",
    "e_nom": "Natural Gas",  # only one e_nom
    "ind_heat_ff_production": "Constraints",
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
    "4hr_battery_storage",
    "8hr_battery_storage",
]


def _round_df(df: pd.DataFrame) -> pd.DataFrame:
    """Converts small values to zero."""
    return df.mask(df.abs() < 1e-8, 0).round(ROUND_TO)


def get_param_names(root: Path, state: str, mode: str) -> dict[str, str]:
    """Gets names and nice_names for the parameter data."""

    assert mode in ("gsa", "ua"), f"Invalid mode: {mode}. Must be 'gsa' or 'ua'."

    data_f = Path(root, "results", state, mode, "parameters.csv")
    df = pd.read_csv(data_f, index_col=0)[["group", "nice_name"]]
    logger.debug(f"Collecting nice names from {data_f}.")
    df = df.drop_duplicates()
    return df.set_index("group")["nice_name"].to_dict()


def get_result_names(root: Path, state: str, mode: str) -> dict[str, str]:
    """Gets names and nice_names for the result data."""

    assert mode in ("gsa", "ua"), f"Invalid mode: {mode}. Must be 'gsa' or 'ua'."

    data_f = Path(root, "results", state, mode, "results.csv")
    df = pd.read_csv(data_f, index_col=0)
    logger.debug(f"Collecting nice names from {data_f}.")
    return df.nice_name.to_dict()


def collect_sa(root: Path, state: str, results: list[str]) -> pd.DataFrame:
    """Collects the SA data from the results directory."""

    data_dir = Path(root, "results", state, "gsa", "SA")
    logger.debug(f"Collecting SA data for {state} from {data_dir}.")

    inputs = [Path(data_dir, f"{x}.csv") for x in results]

    dfs = []
    for f in inputs:
        name = f.stem
        df = pd.read_csv(f, index_col=0).mu_star
        df.name = name
        dfs.append(df)
    df = _round_df(pd.concat(dfs, axis=1))
    df["state"] = state.upper()
    df = df.reset_index(names=["param"])
    return df.set_index(["param", "state"])


def collect_runs(root: Path, state: str, mode: str, results: list[str]) -> pd.DataFrame:
    """Collects the model run results data from the results directory."""

    assert mode in ("gsa", "ua"), f"Invalid mode: {mode}. Must be 'gsa' or 'ua'."

    data_dir = Path(root, "results", state, mode, "results")
    logger.debug(f"Collecting model run results data for {state} from {data_dir}.")

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
    state_upper = state.upper()
    df["state"] = state_upper  # performance warning
    df = df.reset_index(names=["run"])
    return df.set_index(["run", "state"])


def get_empty_sa() -> pd.DataFrame:
    """Returns an empty dataframe for the SA data."""
    return pd.DataFrame(columns=["param", "state"]).set_index(["param", "state"])


def get_empty_run() -> pd.DataFrame:
    """Returns an empty dataframe for the run data."""
    return pd.DataFrame(columns=["param", "state"]).set_index(["param", "state"])


def get_base_params_file(root: Path, state: str) -> pd.DataFrame:
    """Returns the base parameters file for shared between all states."""
    df = pd.read_csv(Path(root, "results", state, "gsa", "parameters.csv"))
    # remove state specific data
    df = df[~((df.name.str.startswith("tct_")) | (df.name == ("emission_limit")))]
    df["state"] = "all"
    return df


def get_state_params(root: Path, state: str) -> pd.DataFrame:
    """Returns the parameters file for the given state and mode."""
    df = pd.read_csv(Path(root, "results", state, "gsa", "parameters.csv"))
    # track state specific data
    df = df[(df.name.str.startswith("tct_")) | (df.name == ("emission_limit"))]
    df["state"] = state.upper()
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
        elif carrier in ["co2", "ch4"]:
            return "Carbon"
        elif carrier.startswith("leakage_"):
            return "Carbon"
        elif carrier.startswith("gwp"):
            return "Carbon"
        elif carrier == "emission_limit":
            return "Carbon"
        elif carrier == "portfolio":
            return "Power"
        elif carrier == "heat_portfolio":
            return "Industry"
        elif carrier in ["imports", "exports"]:
            return "Power"
        elif carrier == "landuse":
            return "System"
        elif carrier in [
            "gas",
            "gas production",
            "gas storage",
            "gas pipeline",
            "gas imports",
            "gas exports",
        ]:
            return "Natural Gas"
        elif carrier == "imports_rec":
            return "Power"
        else:
            raise ValueError(f"Invalid carrier: {carrier}")

    params["attribute_nice_name"] = params.attribute.map(PARAM_ATTRIBUTE_NICE_NAMES)
    params["attribute"] = params.attribute_nice_name.str.replace(" ", "_").str.lower()
    params["sector"] = params.carrier.map(_assign_sector)
    fuel_cost_mask = params.component == "stores_t"
    params.loc[fuel_cost_mask, "sector"] = "Primary Fuel"

    missing = params[(params.attribute.isna()) | (params.attribute_nice_name.isna())]
    if not missing.empty:
        logger.error(
            f"Missing attribute and/or attribute_nice_name: {missing.name.unique()}"
        )
        raise ValueError("Missing attribute and/or attribute_nice_name")

    return params


def correct_params(params: pd.DataFrame) -> pd.DataFrame:
    """Corrects the parameters dataframe for plotting.

    These are just hacky fixes that account for modelling implementation oditities.
    """
    params.loc[params.name == "nuclear_cost", "component"] = "stores_t"
    params.loc[params.name == "nuclear_cost", "sector"] = "Primary Fuel"
    return params


def get_ur_params_expanded(
    ua_params: dict[str, str], params: pd.DataFrame, metadata: dict[str, Any]
) -> dict[str, str]:
    """Expands the CR parameters to be index names, rather than group names."""
    cr_params = {}
    for group, _ in ua_params.items():
        temp = params[params.group == group]
        for row in temp.itertuples():
            if row.name in metadata["parameters"]:
                if "label2" in metadata["parameters"][row.name]:
                    label = metadata["parameters"][row.name]["label2"]
                else:
                    label = metadata["parameters"][row.name]["label"]
            else:
                label = row.nice_name
            cr_params[row.name] = label
    return cr_params


def check_metadata(
    metadata: dict[str, Any], params: pd.DataFrame, results: pd.DataFrame
) -> None:
    """Checks that the metadata exists for all params/results."""
    for param in params.name.unique():
        if param not in metadata["parameters"]:
            raise ValueError(f"Parameter {param} not found in metadata parameters.")
    for group in params.group.unique():
        if group not in metadata["groups"]:
            raise ValueError(f"Group {group} not found in metadata groups.")
    for result in results.name.unique():
        if result not in metadata["results"]:
            raise ValueError(f"Result {result} not found in metadata results.")


if __name__ == "__main__":
    # ensure metadata exists for all params/results
    metadata = json.load(
        open(Path(root, "dashboard", "data", "locked", "metadata.json"))
    )
    for state in STATES:
        try:
            params = pd.read_csv(
                Path(root, "results", state.lower(), "gsa", "parameters.csv")
            )
            results = pd.read_csv(
                Path(root, "results", state.lower(), "ua", "results.csv")
            )
            check_metadata(metadata, params, results)
        except FileNotFoundError:
            continue

    # sensitivity measures

    dfs = []
    empty = True
    for state in STATES:
        state_data = Path(root, "results", state.lower(), "gsa", "SA")
        filtered_data = Path(
            root, "dashboard", "data", "state", state.upper(), "sa.csv"
        )

        if not filtered_data.parent.exists():
            filtered_data.parent.mkdir(parents=True, exist_ok=True)

        if filtered_data.exists():
            sa = pd.read_csv(filtered_data, index_col=[0, 1])
            empty = sa.empty  # do not overwirte if already exists
        elif not state_data.exists():
            logger.warning(f"No gsa model run data for '{state}'")
            sa = get_empty_sa()
        else:
            sa_names = get_result_names(root, state.lower(), "gsa")
            sa = collect_sa(root, state.lower(), sa_names.keys())

        if empty:
            sa.to_csv(filtered_data, index=True)
        dfs.append(sa)

    if not dfs:
        logger.error("No data found.")
        raise ValueError("No GSA data found.")

    df = pd.concat(dfs, axis=0)
    df.to_csv(Path(root, "dashboard", "data", "system", "sa.csv"), index=True)

    # model run results ua

    dfs = []
    empty = True
    for state in STATES:
        state_data = Path(root, "results", state.lower(), "ua", "results")
        filtered_data = Path(
            root, "dashboard", "data", "state", state.upper(), "ua_runs.csv"
        )

        if not filtered_data.parent.exists():
            filtered_data.parent.mkdir(parents=True, exist_ok=True)

        if filtered_data.exists():
            ua = pd.read_csv(filtered_data, index_col=[0, 1])
            empty = ua.empty

        elif not state_data.exists():
            logger.warning(f"No ua model run data for '{state}'")
            ua = get_empty_run()

        else:
            result_names = get_result_names(root, state.lower(), "ua")
            ua = collect_runs(root, state.lower(), "ua", result_names.keys())

        if empty:
            ua.to_csv(filtered_data, index=True)

        ua.to_csv(
            Path(root, "dashboard", "data", "state", state.upper(), "ua_runs.csv"),
            index=True,
        )
        dfs.append(ua)

    if not dfs:
        logger.error("No data found.")
        raise ValueError("No UA Run data found.")

    df = pd.concat(dfs, axis=0)
    df = df.round(2)
    df.to_csv(Path(root, "dashboard", "data", "system", "ua_runs.csv"), index=True)

    # model parameters

    dfs = []
    base_params = pd.DataFrame()
    for state in STATES:
        state_data = Path(root, "results", state.lower(), "gsa")
        if not state_data.exists():
            logger.warning(f"No parameter data for '{state.lower()}'")
        else:
            if base_params.empty:
                base_params = get_base_params_file(root, state.lower())
                dfs.append(base_params)
            state_params = get_state_params(root, state.lower())
            dfs.append(state_params)
    if not dfs:
        logger.error("No data found.")
        raise ValueError("No State data found.")

    params = pd.concat(dfs, axis=0)
    params = assign_parameter_filters(params)
    params = correct_params(params)
    params = params[
        [
            "name",
            "nice_name",
            "group",
            # "group_nice_name",
            "state",
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
    params.to_csv(
        Path(root, "dashboard", "data", "system", "parameters.csv"), index=False
    )

    # get nice names

    all_params = []
    for state in STATES:
        try:
            ua_params = get_param_names(root, state.lower(), "ua")
            all_params.append(ua_params)
        except FileNotFoundError:
            logger.debug(f"No ua nice names for {state}.")
            continue

        ua_params_expanded = get_ur_params_expanded(ua_params, params, metadata)

        with open(
            Path(root, "dashboard", "data", "state", state.upper(), "ua_params.json"),
            "w",
        ) as f:
            json.dump(ua_params_expanded, f, indent=4)

    combined_params = {}
    for params in all_params:
        for param in params:
            if param not in combined_params:
                combined_params[param] = params[param]

    with open(Path(root, "dashboard", "data", "system", "ua_params.json"), "w") as f:
        json.dump(combined_params, f, indent=4)

    # get the sample data
    for state in STATES:
        state_data = Path(root, "results", state.lower(), "ua", "sample_scaled.csv")
        if not state_data.exists():
            logger.warning(f"No sample data for '{state}'")
            continue
        else:
            sample_data = pd.read_csv(state_data)
            sample_data["run"] = sample_data.index
            sample_data["state"] = state.upper()
            sample_data = sample_data.set_index(["run", "state"])
            sample_data.to_csv(
                Path(
                    root, "dashboard", "data", "state", state.upper(), "sample_data.csv"
                ),
                index=True,
            )
