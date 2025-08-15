"""Custom Result."""

from typing import Any
from dash import dcc, html, dash_table

from .utils import (
    DEFAULT_PLOTLY_THEME,
    DEFAULT_HEIGHT,
    DEFAULT_LEGEND,
    get_iso_dropdown_options,
    get_y_label,
    get_emission_limits,
    DEFAULT_2005_EMISSION_LIMIT,
    DEFAULT_2030_EMISSION_LIMIT,
)
from .data import SECTOR_DROPDOWN_OPTIONS_NO_ALL, METADATA
from . import ids as ids
from .styles import DATA_TABLE_STYLE

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
            cr_emission_target_rb(),
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
                multi=True,
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


def cr_emission_target_rb() -> html.Div:
    """Custom Result emission target rb component."""
    return html.Div(
        [
            html.H6("Show Emission Target"),
            dcc.RadioItems(
                id=ids.CR_EMISSION_TARGET_RB,
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},
                ],
                value=True,
                inline=True,
                className="me-3",
                labelStyle={"marginRight": "20px"},
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


def get_stores_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts out stores that will have different units.

    This is ALSO SUUUUUUUPER hacky, but this just NEEDS to get done! :|
    """
    return df[
        (df.result.str.contains("water_heat")) | (df.result.str.contains("battery"))
    ]


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
    data: dict[str, Any],
    nice_names: bool = True,
    marginal: str = None,
    emissions: list[dict[str, Any]] | None = None,
    **kwargs,
) -> go.Figure:
    """UA scatter plot component."""

    if not data:
        logger.debug("No CR scatter plot data found")
        return px.scatter(
            pd.DataFrame(columns=["param", "result"]), x="param", y="result"
        )

    df = _read_serialized_cr_data(data)

    for col in df.columns:
        if col in METADATA["parameters"]:
            id_var = col
            id_var_nice_name = METADATA["parameters"][col]["label"]
            id_var_unit = METADATA["parameters"][col]["unit"]
            break  # only one xlabel

    df_melted = df.melt(id_vars=[id_var], var_name="result", value_name="value")

    ylabels = {}
    for result in df_melted.result:
        ylabels[result] = {}
        if result in METADATA["results"]:
            label_name = (
                "label2" if "label2" in METADATA["results"][result] else "label"
            )
            ylabels[result]["nice_name"] = METADATA["results"][result][label_name]
            ylabels[result]["name"] = result
            ylabels[result]["unit"] = METADATA["results"][result]["unit"]
        else:
            ylabels[result]["nice_name"] = result
            ylabels[result]["name"] = result
            ylabels[result]["unit"] = ""

    df_stores = get_stores_df(df_melted)
    if not df_stores.empty:
        logger.debug("Found stores in CR data")
        df_melted = df_melted[~df_melted.result.isin(df_stores.result)].copy()

    if nice_names:
        df_melted = df_melted.rename(columns={id_var: id_var_nice_name})
        df_melted["result"] = df_melted.result.map(
            {x: y["nice_name"] for x, y in ylabels.items()}
        )

    x_name = id_var_nice_name if nice_names else id_var

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    result_type = kwargs.get("result_type", None)
    ylabel = get_y_label(df_melted, result_type)

    fig = px.scatter(
        df_melted,
        x=x_name,
        y="value",
        marginal_y=marginal,
        color="result",
    )

    fig.update_layout(
        title="",
        xaxis_title=f"{x_name} ({id_var_unit})",
        yaxis_title=ylabel,
        height=DEFAULT_HEIGHT,
        showlegend=True,
        legend=DEFAULT_LEGEND,
        legend_title_text="",
        template=color_theme,
    )

    # SUUUUPER hacky way to account for stores being in different units
    if not df_stores.empty:
        y_label_store = ylabel.replace("(MW)", "(MWh)")

        if nice_names:
            df_stores = df_stores.rename(columns={id_var: id_var_nice_name})
            df_stores["result"] = df_stores.result.map(
                {x: y["nice_name"] for x, y in ylabels.items()}
            )

        scatter_fig = px.scatter(
            df_stores,
            x=x_name,
            y="value",
            color="result",
            marginal_y=marginal,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        # so the stores are clearly different
        if marginal != "histogram":
            scatter_fig.update_traces(marker_symbol="x")

        for trace in scatter_fig.data:
            trace.yaxis = "y2"
            fig.add_trace(trace)

        fig.update_layout(
            yaxis2=dict(
                title=y_label_store,
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

    if emissions:
        emissions_2005, emissions_2030 = get_emission_limits(emissions)

        fig.add_hline(
            y=emissions_2005,
            **DEFAULT_2005_EMISSION_LIMIT,
        )

        fig.add_hline(
            y=emissions_2030,
            **DEFAULT_2030_EMISSION_LIMIT,
        )

    return fig
