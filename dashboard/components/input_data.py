"""Components for input data."""

from typing import Any

import pandas as pd

from dash import dash_table

import logging

logger = logging.getLogger(__name__)


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
    logger.debug(f"Input data table data: {df}")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    logger.debug(f"DataFrame shape: {df.shape}")

    columns = [
        {"name": "name", "id": "name", "type": "text"},
        {"name": "group", "id": "group", "type": "text"},
        {"name": "nice_name", "id": "nice_name", "type": "text"},
        {"name": "iso", "id": "iso", "type": "text"},
        {"name": "component", "id": "component", "type": "text"},
        {"name": "carrier", "id": "carrier", "type": "text"},
        {"name": "attribute", "id": "attribute", "type": "text"},
        {"name": "attribute_nice_name", "id": "attribute_nice_name", "type": "text"},
        {"name": "sector", "id": "sector", "type": "text"},
        {"name": "unit", "id": "unit", "type": "text"},
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
