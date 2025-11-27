"""Component to display uncertainity data for the full system."""

from typing import Any
from dash import dash_table, dcc, html
import plotly.graph_objects as go
import plotly.express as px
from .data import (
    RESULT_SUMMARY_TYPE_DROPDOWN_OPTIONS,
)

import pandas as pd
from . import ids as ids
from .utils import (
    DEFAULT_HEIGHT,
    DEFAULT_LEGEND,
    DEFAULT_PLOTLY_THEME,
    _unflatten_dropdown_options,
)
from .styles import DATA_TABLE_STYLE
import logging

logger = logging.getLogger(__name__)


def ua2_options_block() -> html.Div:
    """UA system options block component."""
    return html.Div(
        [
            ua2_result_type_dropdown(),
            ua2_result_dropdown(),
            ua2_percentile_interval_slider(),
        ],
    )


def ua2_result_type_dropdown() -> html.Div:
    """UA2 result type dropdown component."""
    return html.Div(
        [
            html.H6("Select Result Type"),
            dcc.Dropdown(
                id=ids.UA2_RESULTS_TYPE_DROPDOWN,
                options=RESULT_SUMMARY_TYPE_DROPDOWN_OPTIONS,
                value="cost",
            ),
        ],
    )


def ua2_result_dropdown() -> html.Div:
    """UA result type dropdown component."""
    return html.Div(
        [
            html.H6("Select Result"),
            dcc.Dropdown(
                id=ids.UA2_RESULTS_DROPDOWN,
                options=[],
                value="objective_cost",
            ),
        ],
    )


def ua2_percentile_interval_slider() -> html.Div:
    """UA system slider component."""

    return html.Div(
        [
            html.H6("Percentile Interval:"),
            dcc.RangeSlider(
                id=ids.UA2_INTERVAL_SLIDER,
                min=0,
                max=100,
                value=[0, 100],
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


def apply_nice_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply nice names."""
    logger.debug("Applying nice names")
    ua2_results = _unflatten_dropdown_options(RESULT_SUMMARY_TYPE_DROPDOWN_OPTIONS)
    return df.rename(columns=ua2_results)


def _read_serialized_ua_data(data: dict[str, Any]) -> pd.DataFrame:
    """Read serialized UA data from the dash store."""
    df = pd.DataFrame(data)
    # logger.debug(f"Searlized UA data read: {df}")
    df["run"] = df.run.astype(int)
    return df.set_index("run")


def get_ua2_data_table(
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

    if nice_names:
        df = apply_nice_names(df)

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
        **DATA_TABLE_STYLE,
    )


def filter_ua2_on_result_name(df: pd.DataFrame, result_name: str) -> pd.DataFrame:
    """Filter UA2 data on result type and name."""

    cols = ["run", "state", result_name]
    return df[cols]


def get_ua2_plot(
    data: dict[str, Any],
    nice_names: bool = True,
    result_type: str = "violin",
    **kwargs,
) -> go.Figure:
    """UA2 plot component."""

    if not data:
        logger.debug("No UA plot data found")
        if result_type == "violin":
            return px.violin(
                pd.DataFrame(columns=["result", "value"]), x="result", y="value"
            )
        else:
            return px.box(
                pd.DataFrame(columns=["result", "value"]), x="result", y="value"
            )

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = apply_nice_names(df)

    df = df.reset_index()
    df_melted = df.melt(id_vars=["run", "state"], var_name="result", value_name="value")

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    ylabel = ""

    if result_type == "violin":
        fig = px.violin(
            df_melted,
            x="state",
            y="value",
            labels=dict[str, str](result="", value=ylabel),
        )
    else:
        fig = px.box(
            df_melted,
            x="state",
            y="value",
            labels=dict[str, str](result="", value=ylabel),
        )

    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title=ylabel,
        height=DEFAULT_HEIGHT,
        showlegend=True,
        legend=DEFAULT_LEGEND,
        template=color_theme,
    )

    return fig
