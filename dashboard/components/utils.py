"""Utility functions for the dashboard."""

import pandas as pd
from pathlib import Path
import json
import plotly.colors as pc
import plotly.express as px

# scenarios must follow these names as they are tied to geographic locations
ISOS = {
    "caiso": "California (CAISO)",
    "ercot": "Texas (ERCOT)",
    "isone": "New England (ISO-NE)",
    "miso": "Midcontinent (MISO)",
    "nyiso": "New York (NYISO)",
    "pjm": "PJM Interconnection (PJM)",
    "spp": "Southwest Power Pool (SPP)",
    "northwest": "Northwest",
    "southeast": "Southeast",
    "southwest": "Southwest",
}

# https://www.ferc.gov/power-sales-and-markets/rtos-and-isos
ISO_STATES = {
    "caiso": ["CA"],
    "ercot": ["TX"],
    "isone": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "miso": ["AR", "IL", "IN", "IA", "LA", "MI", "MN", "MO", "MS", "WI"],
    "nyiso": ["NY"],
    "pjm": ["DE", "KY", "MD", "NJ", "OH", "PA", "VA", "WV"],
    "spp": ["KS", "ND", "NE", "OK", "SD"],
    "northwest": ["ID", "MT", "OR", "WA", "WY"],
    "southeast": ["AL", "FL", "GA", "NC", "SC", "TN"],
    "southwest": ["AZ", "CO", "NM", "NV", "UT"],
}

DEFAULT_CONTINOUS_COLOR_SCALE = "pubu"
DEFAULT_DISCRETE_COLOR_SCALE = "Set3"


def _convert_to_dropdown_options(options: dict[str, str]) -> list[dict[str, str]]:
    """Convert a dictionary to a list of dropdown options, sorted alphabetically by label."""
    options_list = [{"label": v, "value": k} for k, v in options.items()]
    return sorted(
        options_list, key=lambda x: x["label"].lower()
    )  # alphabetical label order


def _unflatten_dropdown_options(options: list[dict[str, str]]) -> dict[str, str]:
    """Unflatten a dictionary of options."""
    return {x["value"]: x["label"] for x in options}


def get_iso_dropdown_options() -> list[dict[str, str]]:
    """Get the ISO dropdown options."""
    return _convert_to_dropdown_options(ISOS)


def get_gsa_params_dropdown_options(
    root: str, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the GSA parameters dropdown options."""
    with open(Path(root, "data", "sa_params.json"), "r") as f:
        loaded = json.load(f)
    if flatten:
        return _convert_to_dropdown_options(loaded)
    else:
        return loaded


def get_ua_params_dropdown_options(
    root: str, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the UA parameters dropdown options."""
    with open(Path(root, "data", "ua_params.json"), "r") as f:
        loaded = json.load(f)
    if flatten:
        return _convert_to_dropdown_options(loaded)
    else:
        return loaded


def get_gsa_results_dropdown_options(
    root: str, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the GSA results dropdown options."""
    with open(Path(root, "data", "sa_results.json"), "r") as f:
        loaded = json.load(f)
    if flatten:
        return _convert_to_dropdown_options(loaded)
    else:
        return loaded
    
def get_ua_results_dropdown_options(
    root: str, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the UA results dropdown options."""
    with open(Path(root, "data", "ua_results.json"), "r") as f:
        loaded = json.load(f)
    if flatten:
        return _convert_to_dropdown_options(loaded)
    else:
        return loaded


def _filter_ua_results_on_type(
    loaded: list[dict[str, str]], result_type: str
) -> list[dict[str, str]]:
    """Filter the UA results on type."""
    if result_type == "marginal":
        pass
    elif result_type == "costs":
        pass
    elif result_type == "emissions":
        pass
    elif result_type == "capacity":
        pass
    elif result_type == "generation":
        pass
    else:
        return loaded

def get_ua_results_dropdown_options(
    root: str, result_type: str | None = None, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the UA results dropdown options."""
    with open(Path(root, "data", "ua_results.json"), "r") as f:
        loaded = json.load(f)
    if flatten:
        return _convert_to_dropdown_options(loaded)
    if result_type:
        return _filter_ua_results_on_type(loaded, result_type)
    else:
        return loaded


def get_continuous_color_scale_options() -> list[str]:
    """Get the continuous color scale options."""
    return sorted(pc.named_colorscales())


def get_discrete_color_scale_options() -> list[str]:
    """Get the discrete color scale options."""
    return sorted(
        [
            k
            for k in px.colors.qualitative.__dict__.keys()
            if not k.startswith("__") and not k.endswith("_r")
        ]
    )
