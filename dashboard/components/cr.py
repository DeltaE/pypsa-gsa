"""Custom Result."""

from pathlib import Path
from typing import Any
from dash import dcc, html, dash_table

from .utils import get_iso_dropdown_options
from .data import SECTOR_DROPDOWN_OPTIONS_NO_ALL, METADATA
from . import ids as ids
from .styles import DATA_TABLE_STYLE
import dash_bootstrap_components as dbc

import pandas as pd
import plotly
import plotly.express as px

import logging

logger = logging.getLogger(__name__)


def cr_options_block() -> html.Div:
    """Custom Result options block component."""
    return html.Div(
        [
            cr_iso_dropdown(),
            cr_sector_dropdown(),
            cr_result_dropdown(),
            cr_parameter_dropdown(),
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


def _read_serialized_cr_data(data: dict[str, Any]) -> pd.DataFrame:
    """Read serialized CR data."""
    return pd.DataFrame(data)


def _apply_nice_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply nice names to CR data."""
    mapper = {}
    for col in df.columns:
        try:
            mapper[col] = METADATA["parameters"][col]["label"]
            continue
        except KeyError:
            logger.info(f"No nice name for for value {col} in parameters")
        try:
            mapper[col] = METADATA["results"][col]["label"]
            continue
        except KeyError:
            logger.info(f"No nice name for for value {col} in results")
        mapper[col] = col

    return df.rename(columns=mapper)


def get_cr_data_table(
    data: dict[str, Any], nice_names: bool = True
) -> dash_table.DataTable:
    """CR data table component."""
    if not data:
        logger.debug("No CR data table data found")
        return dash_table.DataTable(
            data=[],
            columns=[],
            style_table={"overflowX": "auto"},
        )

    df = _read_serialized_cr_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    # Format columns for display
    columns = [
        {"name": "run", "id": "run", "type": "numeric", "format": {"specifier": ".0f"}}
    ] + [
        {"name": col, "id": col, "type": "numeric", "format": {"specifier": ".3f"}}
        for col in df.columns
    ]

    df = df.reset_index(names="run")
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=columns,
        **DATA_TABLE_STYLE,
    )
