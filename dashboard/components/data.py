"""Data reader to share across app."""

from pathlib import Path

import pandas as pd
import geopandas as gpd

from .utils import (
    get_cr_data_by_state,
    get_emissions,
    get_metadata,
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
    get_ua_results_dropdown_options,
    get_cr_params_dropdown_options,
    STATES,
)

root = Path(__file__).parent.parent
METADATA = get_metadata(root)

RAW_GSA = pd.read_csv("data/system/sa.csv")
RAW_UA = pd.read_csv("data/system/ua_runs.csv")

RAW_PARAMS = pd.read_csv("data/system/parameters.csv")
param_nice_names = {x: y["label"] for x, y in METADATA["parameters"].items()}
RAW_PARAMS["nice_name"] = RAW_PARAMS.name.map(
    lambda x: param_nice_names[x] if x in param_nice_names else x
)
group_nice_names = {x: y["label"] for x, y in METADATA["groups"].items()}
RAW_PARAMS["group_nice_name"] = RAW_PARAMS.group.map(
    lambda x: group_nice_names[x] if x in group_nice_names else x
)

# ISO_SHAPE = gpd.read_file("data/locked/iso.geojson")
STATE_SHAPE_ACTUAL = gpd.read_file("data/locked/states.geojson")
STATE_SHAPE_HEX = gpd.read_file("data/locked/states_hex.geojson")

GSA_PARM_OPTIONS = get_gsa_params_dropdown_options(METADATA)
GSA_RESULT_OPTIONS = get_gsa_results_dropdown_options(METADATA, list(RAW_GSA))

# UA_PARAM_OPTIONS = get_ua_params_dropdown_options(METADATA) # need state
UA_RESULT_OPTIONS = get_ua_results_dropdown_options(METADATA)

CR_PARAM_OPTIONS = {
    state: get_cr_params_dropdown_options(root, state) for state in STATES
}

CR_DATA = {state: get_cr_data_by_state(root, state) for state in STATES}

SECTOR_DROPDOWN_OPTIONS = sorted(
    [{"label": y, "value": x} for x, y in METADATA["nice_names"]["sector"].items()],
    key=lambda x: x["label"],
)
SECTOR_DROPDOWN_OPTIONS_NO_ALL = [
    x for x in SECTOR_DROPDOWN_OPTIONS if x["value"] != "all"
]
SECTOR_DROPDOWN_OPTIONS_ALL = [
    x for x in SECTOR_DROPDOWN_OPTIONS if x["value"] == "all"
]
SECTOR_DROPDOWN_OPTIONS_SYSTEM = [
    x for x in SECTOR_DROPDOWN_OPTIONS if x["value"] == "system"
]
SECTOR_DROPDOWN_OPTIONS_SYSTEM_POWER_NG = [
    x
    for x in SECTOR_DROPDOWN_OPTIONS
    if x["value"] in ["system", "power", "natural_gas"]
]
SECTOR_DROPDOWN_OPTIONS_IDV = sorted(
    [
        x
        for x in SECTOR_DROPDOWN_OPTIONS
        if x["value"] in ["power", "industry", "service", "transport", "natural_gas"]
    ],
    key=lambda x: x["label"],
)
SECTOR_DROPDOWN_OPTIONS_TRADE = sorted(
    [x for x in SECTOR_DROPDOWN_OPTIONS if x["value"] in ["power", "natural_gas"]],
    key=lambda x: x["label"],
)

RESULT_TYPE_DROPDOWN_OPTIONS = sorted(
    [{"label": y, "value": x} for x, y in METADATA["nice_names"]["results"].items()],
    key=lambda x: x["label"],
)

EMISSIONS = get_emissions(root)
