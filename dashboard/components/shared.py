"""Shared components for the dashboard."""

from dash import html, dcc
import pandas as pd
import plotly.colors as pc
from . import ids as ids
from .utils import get_iso_dropdown_options

ISO_OPTIONS = get_iso_dropdown_options()

import logging

logger = logging.getLogger(__name__)


def iso_options_block(*args: pd.DataFrame | None) -> html.Div:
    """ISO options block component."""

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

    logger.debug(f"Loaded isos: {options}")

    default = [x["value"] for x in options] if options else None

    logger.debug(f"Default ISO options: {default}")

    return html.Div(
        [
            iso_dropdown(options, default),
        ],
    )


def iso_dropdown(options: list[str], default: list[str] | None = None) -> html.Div:
    """ISO dropdown component."""

    return html.Div(
        [
            html.H6("Select ISO(s)"),
            dcc.Dropdown(
                id=ids.ISO_DROPDOWN,
                options=options,
                value=default if default else options[0],
                multi=True,
            ),
        ],
        className="dropdown-container",
    )


def plotting_options_block() -> html.Div:
    """Plotting options block component."""

    return html.Div(
        [
            plotting_type_dropdown(),
            color_scale_dropdown(),
        ],
        className="dropdown-container",
    )


def color_scale_dropdown() -> html.Div:
    """Color scale dropdown component."""

    return html.Div(
        [
            html.H6("Select Color Scale"),
            dcc.Dropdown(
                id=ids.COLOR_DROPDOWN,
                options=sorted(pc.named_colorscales()),
                value="pubu",
                multi=False,
            ),
        ],
        className="dropdown-container",
    )


def plotting_type_dropdown() -> html.Div:
    """Plotting type dropdown component."""

    return html.Div(
        [
            html.H6("Select Data Visualization"),
            dcc.Dropdown(
                id=ids.PLOTTING_TYPE_DROPDOWN, options=[], value="heatmap", multi=False
            ),
        ],
        className="dropdown-container",
    )
