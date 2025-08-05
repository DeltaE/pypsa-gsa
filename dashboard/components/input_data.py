"""Components for input data."""

from typing import Any

import pandas as pd

from .styles import BUTTON_STYLE, DATA_TABLE_STYLE
from . import ids as ids
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

import logging

logger = logging.getLogger(__name__)


def input_data_options_block() -> html.Div:
    """Input data options block component."""
    return html.Div(
        [
            input_data_attribute_dropdown(),
            input_data_sector_dropdown(),
            input_data_remove_filters_button(),
        ],
    )


def input_data_attribute_dropdown() -> html.Div:
    """Input data attribute dropdown component."""
    return html.Div(
        [
            html.H6("Select Attribute"),
            dcc.Dropdown(
                id=ids.INPUT_DATA_ATTRIBUTE_DROPDOWN,
                options={},
                value="",
                disabled=False,
            ),
        ],
    )


def input_data_sector_dropdown() -> html.Div:
    """Input data sector type dropdown component."""
    return html.Div(
        [
            html.H6("Select Group"),
            dcc.Dropdown(
                id=ids.INPUT_DATA_SECTOR_DROPDOWN,
                options={},
                value="",
                disabled=True,
            ),
        ],
    )


def input_data_remove_filters_button() -> html.Div:
    """Input data remove filters component."""
    return html.Div(
        [
            dbc.Button(
                "Remove All",
                id=ids.INPUT_DATA_REMOVE_FILTERS,
                **BUTTON_STYLE,
            ),
        ],
    )


def _read_serialized_data(data: dict[str, Any]) -> pd.DataFrame:
    """Read serialized inputs data."""
    df = pd.DataFrame(data)
    return df


def get_inputs_data_table(data: dict[str, Any]) -> dash_table.DataTable:
    """Inputs data table component."""
    if not data:
        logger.debug("No input data table data found")
        return dash_table.DataTable(
            data=[],
            columns=[],
            style_table={"overflowX": "auto"},
        )

    df = _read_serialized_data(data)

    columns = [
        {"name": "name", "id": "name", "type": "text"},
        {"name": "nice_name", "id": "name", "type": "text"},
        {"name": "group", "id": "group", "type": "text"},
        {"name": "group_nice_name", "id": "group_nice_name", "type": "text"},
        {"name": "iso", "id": "iso", "type": "text"},
        {"name": "component", "id": "component", "type": "text"},
        {"name": "carrier", "id": "carrier", "type": "text"},
        {"name": "attribute", "id": "attribute", "type": "text"},
        {"name": "attribute_nice_name", "id": "attribute_nice_name", "type": "text"},
        {"name": "sector", "id": "sector", "type": "text"},
        {"name": "unit", "id": "unit", "type": "text"},
        {"name": "range", "id": "range", "type": "text"},
        {
            "name": "min_value",
            "id": "min_value",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        {
            "name": "max_value",
            "id": "max_value",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        {"name": "source", "id": "source", "type": "text"},
        {"name": "notes", "id": "notes", "type": "text"},
    ]

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=columns,
        **DATA_TABLE_STYLE,
    )


def get_input_data_barchart(data: dict[str, Any], **kwargs: Any) -> dcc.Graph:
    """Input data barchart component."""
    if not data:
        logger.debug("No input data barchart data found")
        return dcc.Graph(
            id=ids.INPUT_DATA_BAR_CHART,
            figure=px.bar(
                pd.DataFrame(columns=["param", "value"]), x="param", y="value"
            ),
        )

    df = _read_serialized_data(data)

    if len(df.sector.unique()) > 1 or len(df.attribute.unique()) > 1:
        logger.debug("More filtering required for barchart")
        return dcc.Graph(
            id=ids.INPUT_DATA_BAR_CHART,
            figure=px.bar(
                pd.DataFrame(columns=["param", "value"]), x="param", y="value"
            ),
        )

    # seperate figures by absolute/percent ranges
    figures = []

    for range_type in df.range.unique():
        df_range = df[df.range == range_type]
        units = df_range.unit.unique()

        if range_type == "absolute":
            for unit in units:
                df_range_unit = df_range[df_range.unit == unit]
                x = df_range_unit.nice_name
                y_start = df_range_unit.min_value
                y_end = df_range_unit.max_value
                heights = y_end - y_start

                fig = go.Figure()
                fig.add_trace(go.Bar(x=x, y=heights, base=y_start))

                title = "Absolute Range"
                yaxis_title = unit
                xaxis_title = ""

                fig.update_layout(
                    title=title,
                    yaxis_title=yaxis_title,
                    xaxis_title=xaxis_title,
                )

                figures.append(dcc.Graph(figure=fig))

        elif range_type == "percent":
            # should only ever be percent
            if len(units) > 1:
                logger.debug(f"Non-unique units of {units} for {range_type}")
                return dcc.Graph(
                    id=ids.INPUT_DATA_BAR_CHART,
                    figure=px.bar(
                        pd.DataFrame(columns=["param", "value"]), x="param", y="value"
                    ),
                )

            x = df_range.nice_name
            y_start = df_range.min_value
            y_end = df_range.max_value
            heights = y_end - y_start

            fig = go.Figure()

            fig.add_trace(go.Bar(x=x, y=heights, base=y_start))

            title = "Percent Range"
            yaxis_title = "%"
            xaxis_title = ""

            fig.update_layout(
                title=title,
                yaxis_title=yaxis_title,
                xaxis_title=xaxis_title,
            )

            figures.append(dcc.Graph(figure=fig))

        else:
            logger.debug(f"Unknown range type: {range_type}")
            return dcc.Graph(
                id=ids.INPUT_DATA_BAR_CHART,
                figure=px.bar(
                    pd.DataFrame(columns=["param", "value"]), x="param", y="value"
                ),
            )

    return html.Div(
        figures,
        id=ids.INPUT_DATA_BAR_CHART,
    )
