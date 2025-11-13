"""Shared components for the dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.colors as pc
from . import ids as ids
from .utils import get_state_dropdown_options
from .styles import BUTTON_STYLE

import logging

STATE_OPTIONS = get_state_dropdown_options()

logger = logging.getLogger(__name__)


def state_options_block(*args: pd.DataFrame | None) -> html.Div:
    """STATE options block component."""

    # only allow available states to be selected if data is loaded
    if not args:
        loaded_states = [x["value"] for x in STATE_OPTIONS]
    else:
        loaded_states = []
        for df in args:
            if "state" in df.columns:
                loaded_states.extend(df.state.unique())
        if not loaded_states:
            loaded_states = [x["value"] for x in STATE_OPTIONS]
        else:
            loaded_states = list(set(loaded_states))

    options = [x for x in STATE_OPTIONS if x["value"] in loaded_states]

    logger.debug(f"Loaded states: {options}")

    default = [x["value"] for x in options] if options else None

    logger.debug(f"Default STATE options: {default}")

    return html.Div(
        [
            state_dropdown(options, multi=True, default=default),
        ],
    )


def state_dropdown(options: list[str], multi: bool = True, default: list[str] | None = None) -> html.Div:
    """STATE dropdown component."""

    return html.Div(
        [
            html.H6("Select State(s)"),
            dcc.Dropdown(
                id=ids.STATE_DROPDOWN,
                options=options,
                value=default if default else options[0],
                multi=multi,
            ),
            html.Div(
                [
                    dbc.Button(
                        "Select All",
                        id=ids.STATES_SELECT_ALL,
                        **BUTTON_STYLE,
                    ),
                    dbc.Button(
                        "Remove All",
                        id=ids.STATES_REMOVE_ALL,
                        **BUTTON_STYLE,
                    ),
                ],
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
            html.H6("Select Color Theme"),
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
