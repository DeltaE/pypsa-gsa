"""Utility functions for the dashboard."""

import pandas as pd
from pathlib import Path
import json

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
    "southwest": ["AZ", "CO", "NM", "NV", "UT"]
}

root = Path(__file__).parent.parent


def get_iso_dropdown() -> dict[str, str]:
    """Get the ISO dropdown options."""
    return ISOS


def get_gsa_params_dropdown() -> dict[str, str]:
    """Get the GSA parameters dropdown options."""
    with open(Path(root, "dashboard", "data", "gsa_params.json"), "r") as f:
        loaded = json.load(f)
    return loaded


def get_ua_params_dropdown() -> dict[str, str]:
    """Get the UA parameters dropdown options."""
    with open(Path(root, "dashboard", "data", "ua_params.json"), "r") as f:
        loaded = json.load(f)
    return loaded


def get_gsa_results_dropdown() -> dict[str, str]:
    """Get the GSA results dropdown options."""
    with open(Path(root, "dashboard", "data", "gsa_results.json"), "r") as f:
        loaded = json.load(f)
    return loaded


def get_ua_results_dropdown() -> dict[str, str]:
    """Get the UA results dropdown options."""
    with open(Path(root, "dashboard", "data", "gsa_results.json"), "r") as f:
        loaded = json.load(f)
    return loaded
