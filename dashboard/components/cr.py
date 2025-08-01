"""Custom Result."""

from pathlib import Path
from dash import dcc, html, dash_table

from .utils import (
    get_iso_dropdown_options,
    get_ua_results_dropdown_options,
    SECTOR_DROPDOWN_OPTIONS_NO_ALL,
)
from . import ids as ids
from .styles import BUTTON_STYLE, DATA_TABLE_STYLE
import dash_bootstrap_components as dbc

import pandas as pd
import plotly
import plotly.express as px

import logging

logger = logging.getLogger(__name__)

root = Path(__file__).parent.parent

RESULT_OPTIONS = get_ua_results_dropdown_options(root)  # ua gives same results


def cr_options_block() -> html.Div:
    """Custom Result options block component."""
    return html.Div(
        [
            cr_iso_dropdown(),
            cr_sector_dropdown(),
            cr_parameter_dropdown(),
            cr_result_dropdown(),
            cr_percentile_interval_slider(),
        ],
    )


def cr_iso_dropdown() -> html.Div:
    """Custom Result ISO dropdown component."""
    isos = get_iso_dropdown_options()
    return html.Div(
        [
            html.H6("Select ISO"),
            dcc.Dropdown(
                id=ids.CR_ISO_DROPDOWN,
                options=isos,
            ),
        ],
    )


def cr_sector_dropdown() -> html.Div:
    """Custom Result sector dropdown component."""
    return html.Div(
        [
            html.H6("Select Sector"),
            dcc.Dropdown(
                id=ids.CR_SECTOR_DROPDOWN,
                options=SECTOR_DROPDOWN_OPTIONS_NO_ALL,
            ),
        ],
    )


def cr_parameter_dropdown() -> html.Div:
    """Custom Result parameter dropdown component."""
    return html.Div(
        [
            html.H6("Select Parameter (x-axis)"),
            dcc.Dropdown(
                id=ids.CR_PARAMETER_DROPDOWN,
                options=[],
            ),
        ],
    )


def cr_result_dropdown() -> html.Div:
    """Custom Result dropdown component."""

    return html.Div(
        [
            html.H6("Select Result (y-axis)"),
            dcc.Dropdown(
                id=ids.CR_RESULT_TYPE_DROPDOWN,
                value="",
            ),
            dcc.Dropdown(
                id=ids.CR_RESULT_DROPDOWN,
                value="",
            ),
        ],
    )


def cr_percentile_interval_slider() -> html.Div:
    """Custom Result slider component."""

    return html.Div(
        [
            html.H6("Percentile Interval:"),
            dcc.RangeSlider(
                id=ids.CR_INTERVAL_SLIDER,
                min=0,
                max=100,
                value=[1, 99],
                step=1,
                included=False,
                marks={x: str(x) for x in range(0, 101, 25)},
                tooltip={
                    "placement": "bottom",
                    "always_visible": False,
                    "template": "{value}%",
                },
            ),
        ],
    )
