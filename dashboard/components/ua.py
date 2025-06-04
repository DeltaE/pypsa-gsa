"""Component to display uncertainity data."""

from typing import Any
from dash import dcc, html, dash_table
import pandas as pd
from pathlib import Path
from .utils import (
    get_ua_params_dropdown_options,
    get_ua_results_dropdown_options,
)
from . import ids as ids
from .utils import _unflatten_dropdown_options

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


def _filter_ua_on_result_sector(df: pd.DataFrame, result_sector: str) -> pd.DataFrame:
    """Filter UA data on result sector."""
    return df


def _filter_ua_on_result_type(df: pd.DataFrame, result_type: str) -> pd.DataFrame:
    """Filter UA data on result type."""
    if result_type == "costs":
        cols = [x for x in df.columns if "objective_" in x]
    else:
        cols = list(df.columns)

    if not cols:
        logger.debug(f"No columns found for result type {result_type}")
        return pd.DataFrame(index=df.index)
    else:
        return df.set_index("run")[cols].reset_index()


def _read_serialized_ua_data(data: dict[str, Any]) -> pd.DataFrame:
    """Read serialized UA data from the dash store."""
    df = pd.DataFrame(data)
    df["run"] = df.run.astype(int)
    return df.set_index("run")


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
        logger.debug("Applying nice names to UA data table")
        ua_results = _unflatten_dropdown_options(UA_RESULT_OPTIONS)
        df = df.rename(columns=ua_results)

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
