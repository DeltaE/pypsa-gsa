"""Shared components for the dashboard."""

from dash import html, dcc
import pandas as pd
from . import ids as ids
from .utils import get_iso_dropdown_options

ISO_OPTIONS = get_iso_dropdown_options()

def iso_dropdown(*args: pd.DataFrame | None) -> html.Div:
    """ISO dropdown component."""
    
    # only allow available isos to be selected if data is loaded
    if not args:
        loaded_isos = [x["value"] for x in ISO_OPTIONS]
    else:
        loaded_isos = []
        for df in args:
            if "iso" in df.columns:
                loaded_isos.extend(df.iso.unique())
        if not loaded_isos:
            loaded_isos = [x["value"] for x in ISO_OPTIONS]
        else:
            loaded_isos = list(set(loaded_isos))
            
    options = [x for x in ISO_OPTIONS if x["value"] in loaded_isos]
    
    return html.Div(
        [
            html.Label("ISO"),
            dcc.Dropdown(
                id=ids.ISO_DROPDOWN,
                options=options,
                value=loaded_isos if loaded_isos else None,
                multi=True
            ),
        ],
        className="dropdown-container",
    )