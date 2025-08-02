"""Custom Result."""

from pathlib import Path
from typing import Any
from dash import dcc, html, dash_table
import numpy as np

from .utils import (
    DEFAULT_OPACITY,
    DEFAULT_PLOTLY_THEME,
    DEFAULT_HEIGHT,
    DEFAULT_LEGEND,
    get_iso_dropdown_options,
)
from .data import SECTOR_DROPDOWN_OPTIONS_NO_ALL, METADATA
from . import ids as ids
from .styles import DATA_TABLE_STYLE
from scipy.stats import gaussian_kde

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
                multi=False,
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


def get_cr_scatter_plot(
    data: dict[str, Any], nice_names: bool = True, marginal: str = None, **kwargs
) -> go.Figure:
    """UA scatter plot component."""

    if not data:
        logger.debug("No CR scatter plot data found")
        return px.scatter(
            pd.DataFrame(columns=["param", "result"]), x="param", y="result"
        )

    df = _read_serialized_cr_data(data)

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)

    xlabel = {}
    ylabel = {}
    for col in df.columns:
        try:
            xlabel["nice_name"] = METADATA["parameters"][col]["label"]
            xlabel["name"] = col
            xlabel["unit"] = METADATA["parameters"][col]["unit"]
            continue
        except KeyError:
            logger.debug(f"{col} not in parameters")
        try:
            label_name = "label2" if "label2" in METADATA["results"][col] else "label"
            ylabel["nice_name"] = METADATA["results"][col][label_name]
            ylabel["name"] = col
            ylabel["unit"] = METADATA["results"][col]["unit"]
            continue
        except KeyError:
            logger.info(f"No nice name for for value {col} in results")
        logger.error(f"No metadata found for {col}")
        return px.scatter(
            pd.DataFrame(columns=["param", "result"]), x="param", y="result"
        )

    mapper = {xlabel["name"]: xlabel["nice_name"], ylabel["name"]: ylabel["nice_name"]}
    if nice_names:
        df = df.rename(columns=mapper)

    x_name = xlabel["nice_name"] if nice_names else xlabel["name"]
    y_name = ylabel["nice_name"] if nice_names else ylabel["name"]

    fig = px.scatter(
        df,
        x=x_name,
        y=y_name,
        marginal_y=marginal,
    )

    fig.update_layout(
        title="",
        xaxis_title=f"{x_name} ({xlabel['unit']})",
        yaxis_title=f"{y_name} ({ylabel['unit']})",
        height=DEFAULT_HEIGHT,
        showlegend=True,
        legend=DEFAULT_LEGEND,
        template=color_theme,
    )

    return fig
