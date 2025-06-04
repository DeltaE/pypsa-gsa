"""Component to display uncertainity data."""

from typing import Any
from dash import dcc, html, dash_table
import pandas as pd
from pathlib import Path
from .utils import (
    DEFAULT_DISCRETE_COLOR_SCALE,
    get_ua_params_dropdown_options,
    get_ua_results_dropdown_options,
)
from . import ids as ids
from .utils import _unflatten_dropdown_options
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde


import logging

logger = logging.getLogger(__name__)

root = Path(__file__).parent.parent

UA_PARM_OPTIONS = get_ua_params_dropdown_options(root)
UA_RESULT_OPTIONS = get_ua_results_dropdown_options(root)

SECTOR_DROPDOWN_OPTIONS = [
    {"label": "All", "value": "all"},
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


def ua_options_block() -> html.Div:
    """UA options block component."""
    return html.Div(
        [
            ua_result_type_dropdown(),
            ua_result_sector_dropdown(),
            ua_percentile_interval_slider(),
        ],
    )


def ua_result_type_dropdown() -> html.Div:
    """UA result type dropdown component."""
    return html.Div(
        [
            html.H6("Select Result Type"),
            dcc.Dropdown(
                id=ids.UA_RESULTS_TYPE_DROPDOWN,
                options=RESULT_TYPE_DROPDOWN_OPTIONS,
                value="costs",
            ),
        ],
    )


def ua_result_sector_dropdown() -> html.Div:
    """UA result type dropdown component."""
    return html.Div(
        [
            html.H6("Select Sector"),
            dcc.Dropdown(
                id=ids.UA_RESULTS_SECTOR_DROPDOWN,
                options=SECTOR_DROPDOWN_OPTIONS,
                value="all",
            ),
        ],
    )


def ua_percentile_interval_slider() -> html.Div:
    """GSA slider component."""

    return html.Div(
        [
            html.H6("Percentile Interval:"),
            dcc.RangeSlider(
                id=ids.UA_INTERVAL_SLIDER,
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


def _filter_ua_on_result_sector(df: pd.DataFrame, result_sector: str) -> pd.DataFrame:
    """Filter UA data on result sector."""
    return df


def _filter_ua_on_result_type(df: pd.DataFrame, result_type: str) -> pd.DataFrame:
    """Filter UA data on result type."""
    if result_type == "costs":
        cols = [x for x in df.columns if "objective_" in x]
    elif result_type == "marginal_costs":
        cols = [x for x in df.columns if "marginal_cost_" in x]
    elif result_type == "emissions":
        cols = list(df.columns)
    elif result_type == "new_capacity":
        cols = list(df.columns)
    elif result_type == "total_capacity":
        cols = list(df.columns)
    else:
        cols = list(df.columns)

    if not cols:
        logger.debug(f"No columns found for result type {result_type}")
        return pd.DataFrame(index=df.index)
    else:
        return df.set_index("run")[cols].reset_index()


def remove_ua_outliers(df: pd.DataFrame, interval: list[int]) -> pd.DataFrame:
    """Replace UA outliers with NaN values."""
    # intervals defined as ints, but pandas quantile expects [0,1]
    interval_low, interval_high = (
        round(min(interval) / 100, 2),
        round(max(interval) / 100, 2),
    )
    logger.debug(
        f"Removing UA outliers with interval {interval_low} and {interval_high}"
    )

    logger.debug(f"Removing UA outliers from: {df}")
    df_out = df.copy()
    for col in df.columns:
        if col == "run":  # shouldnt really happen, but just in case
            df_out[col] = df[col]
        else:
            lower_bound = df[col].quantile(interval_low)
            upper_bound = df[col].quantile(interval_high)
            logger.debug(
                f"Removing UA outliers from {col} with interval {interval_low} and {interval_high}"
            )
            df_out[col] = df[col].where(
                (df[col] >= lower_bound) & (df[col] <= upper_bound)
            )

    return df_out


def _read_serialized_ua_data(data: dict[str, Any]) -> pd.DataFrame:
    """Read serialized UA data from the dash store."""
    df = pd.DataFrame(data)
    logger.debug(f"Searlized UA data read: {df}")
    df["run"] = df.run.astype(int)
    return df.set_index("run")


def _apply_nice_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply nice names to UA data table."""
    logger.debug("Applying nice names to UA data table")
    ua_results = _unflatten_dropdown_options(UA_RESULT_OPTIONS)
    return df.rename(columns=ua_results)


def filter_ua_on_result_sector_and_type(
    df: pd.DataFrame, result_sector: str, result_type: str
) -> pd.DataFrame:
    """Filter UA data on result sector and type."""
    filtered_on_sector = _filter_ua_on_result_sector(df, result_sector)
    filtered_on_sector_and_type = _filter_ua_on_result_type(
        filtered_on_sector, result_type
    )
    return filtered_on_sector_and_type


def get_ua_data_table(
    data: dict[str, Any], nice_names: bool = True
) -> dash_table.DataTable:
    """UA data table component."""
    if not data:
        logger.debug("No UA data table data found")
        return dash_table.DataTable(
            data=[],
            columns=[],
            style_table={"overflowX": "auto"},
        )

    df = _read_serialized_ua_data(data)

    logger.debug(f"UA data table: {df}")

    if nice_names:
        df = _apply_nice_names(df)

    # Format columns for display
    columns = [
        {"name": "run", "id": "run", "type": "numeric", "format": {"specifier": ".0f"}}
    ] + [
        {"name": col, "id": col, "type": "numeric", "format": {"specifier": ".3f"}}
        for col in df.columns
        if col != "param"
    ]

    df = df.reset_index()

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=columns,
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
            "border": "1px solid black",
        },
        style_data={"border": "1px solid lightgrey"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
        page_size=50,
        sort_action="native",
        filter_action="native",
        sort_mode="multi",
        export_format="csv",
        export_headers="display",
    )


def get_ua_scatter_plot(
    data: dict[str, Any], nice_names: bool = True, **kwargs
) -> go.Figure:
    """UA scatter plot component."""

    if not data:
        logger.debug("No UA scatter plot data found")
        return px.scatter(pd.DataFrame(), x="run", y="value")

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = df.melt(id_vars=["run"], var_name="result", value_name="value")

    # drop outlier data
    df_melted = df_melted.dropna(subset=["value"])

    color_scale = kwargs.get("color_scale", DEFAULT_DISCRETE_COLOR_SCALE)
    logger.debug(f"UA scatter plot color scale: {color_scale}")

    fig = px.scatter(
        df_melted,
        x="run",
        y="value",
        color="result",
        labels=dict(run="Run Number", value="Value", result="Result Type"),
    )

    fig.update_layout(
        title="Uncertainty Analysis Results",
        xaxis_title="Run Number",
        yaxis_title="Value",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def get_ua_barchart(
    data: dict[str, Any], nice_names: bool = True, **kwargs
) -> go.Figure:
    """UA barchart component showing mean values with 95% confidence intervals for each result category."""
    if not data:
        logger.debug("No UA barchart data found")
        return px.bar(pd.DataFrame(), x="result", y="value")

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = df.melt(id_vars=["run"], var_name="result", value_name="value")

    # Remove NaN values
    df_melted = df_melted.dropna(subset=["value"])

    # Calculate mean and std for each result category
    stats_df = (
        df_melted.groupby("result")
        .agg(value=("value", "mean"), std=("value", "std"))
        .reset_index()
    )

    fig = px.bar(
        stats_df,
        x="result",
        y="value",
        error_y="std",
        labels=dict(result="", value="Value"),
    )

    # Update layout
    fig.update_layout(
        title="Uncertainty Analysis (Mean ± 95% CI)",
        showlegend=False,
        height=600,
        bargap=0.2,
        xaxis=dict(tickangle=45),
    )

    return fig


def get_ua_histogram(
    data: dict[str, Any], nice_names: bool = True, **kwargs
) -> go.Figure:
    """UA histogram component with overlaid probability density functions."""
    if not data:
        logger.debug("No UA histogram data found")
        return px.histogram(pd.DataFrame(), x="value")

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = df.melt(id_vars=["run"], var_name="result", value_name="value")

    # base histogram
    fig = px.histogram(
        df_melted,
        x="value",
        color="result",
        labels=dict(value="Value", result="Result Type"),
        opacity=0.7,
    )

    # Add secondary y-axis for density
    fig.update_layout(
        yaxis2=dict(
            title="Probability Density",
            overlaying="y",
            side="right",
            showgrid=False,
        )
    )

    # PDF for each result
    for result_type in df_melted["result"].unique():
        result_data = df_melted[df_melted["result"] == result_type]["value"].dropna()

        kde = gaussian_kde(result_data)
        x_range = np.linspace(result_data.min(), result_data.max(), 100)
        y_range = kde(x_range)

        # Scale the density to match the histogram scale
        # Get the histogram bin width to scale the density appropriately
        hist_data = fig.data[0]  # Get first histogram trace
        bin_width = (hist_data.x[1] - hist_data.x[0]) if len(hist_data.x) > 1 else 1
        total_count = len(result_data)
        y_range = y_range * total_count * bin_width * (-1) # bin width is negative

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                name=f"{result_type} (PDF)",
                line=dict(width=2),
                showlegend=True,
                yaxis="y2",  # secondary y-axis
            )
        )

    fig.update_layout(
        title="Uncertainty Analysis Distribution",
        xaxis_title="Value",
        yaxis_title="Count",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
