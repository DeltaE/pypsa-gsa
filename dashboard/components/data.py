"""Data reader to share across app."""

from pathlib import Path

import pandas as pd
import geopandas as gpd

from .utils import (
    get_metadata,
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
    get_ua_params_dropdown_options,
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

# UA_PARAM_OPTIONS = get_ua_params_dropdown_options(METADATA)
UA_RESULT_OPTIONS = get_ua_results_dropdown_options(METADATA)

CR_PARAM_OPTIONS = {iso: get_cr_params_dropdown_options(root, iso) for iso in ISOS}

SECTOR_DROPDOWN_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "System", "value": "system"},
    {"label": "Power", "value": "power"},
    {"label": "Industry", "value": "industry"},
    {"label": "Service", "value": "service"},
    {"label": "Transportation", "value": "transport"},
]
SECTOR_DROPDOWN_OPTIONS_NO_ALL = [
    {"label": "System", "value": "system"},
    {"label": "Power", "value": "power"},
    {"label": "Industry", "value": "industry"},
    {"label": "Service", "value": "service"},
    {"label": "Transportation", "value": "transport"},
]
SECTOR_DROPDOWN_OPTIONS_ALL = [
    {"label": "All", "value": "all"},
]
SECTOR_DROPDOWN_OPTIONS_IDV = [
    {"label": "Power", "value": "power"},
    {"label": "Industry", "value": "industry"},
    {"label": "Service", "value": "service"},
    {"label": "Transportation", "value": "transport"},
]
RESULT_TYPE_DROPDOWN_OPTIONS = [
    {"label": "Costs", "value": "costs"},
    {"label": "Marginal Costs", "value": "marginal_costs"},
    {"label": "Emissions", "value": "emissions"},
    {"label": "New Capacity", "value": "new_capacity"},
    {"label": "Total Capacity", "value": "total_capacity"},
    {"label": "Generation", "value": "generation"},
]
