"""Data reader to share across app."""

from pathlib import Path

import pandas as pd
import geopandas as gpd

from .utils import (
    get_cr_data_by_iso,
    get_metadata,
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
    get_ua_results_dropdown_options,
    get_cr_params_dropdown_options,
    ISOS,
)

root = Path(__file__).parent.parent
METADATA = get_metadata(root)

RAW_GSA = pd.read_csv("data/system/sa.csv")
RAW_UA = pd.read_csv("data/system/ua_runs.csv")
RAW_PARAMS = pd.read_csv("data/system/parameters.csv")
ISO_SHAPE = gpd.read_file("data/locked/iso.geojson")

GSA_PARM_OPTIONS = get_gsa_params_dropdown_options(METADATA)
GSA_RESULT_OPTIONS = get_gsa_results_dropdown_options(METADATA, list(RAW_GSA))

# UA_PARAM_OPTIONS = get_ua_params_dropdown_options(METADATA) # need iso
UA_RESULT_OPTIONS = get_ua_results_dropdown_options(METADATA)

CR_PARAM_OPTIONS = {iso: get_cr_params_dropdown_options(root, iso) for iso in ISOS}

CR_DATA = {iso: get_cr_data_by_iso(root, iso) for iso in ISOS}

SECTOR_DROPDOWN_OPTIONS = [
    {"label": y, "value": x} for x, y in METADATA["nice_names"]["sector"].items()
]
SECTOR_DROPDOWN_OPTIONS_NO_ALL = [
    x for x in SECTOR_DROPDOWN_OPTIONS if x["value"] != "all"
]
SECTOR_DROPDOWN_OPTIONS_ALL = [
    x for x in SECTOR_DROPDOWN_OPTIONS if x["value"] == "all"
]
SECTOR_DROPDOWN_OPTIONS_IDV = [
    x
    for x in SECTOR_DROPDOWN_OPTIONS
    if x["value"] in ["power", "industry", "service", "transport"]
]

RESULT_TYPE_DROPDOWN_OPTIONS = [
    {"label": y, "value": x} for x, y in METADATA["nice_names"]["results"].items()
]
