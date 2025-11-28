"""Component to display uncertainity data for the full system."""

from typing import Any
from dash import dash_table, dcc, html
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from .data import (
    RESULT_SUMMARY_TYPE_DROPDOWN_OPTIONS,
)
import dash_bootstrap_components as dbc
import pandas as pd
from . import ids as ids
from .utils import (
    DEFAULT_DISCRETE_COLOR_SCALE,
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
    metadata: dict[str, Any] = None,
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

    if metadata:
        value_name = df.columns[-1]
        ylabel = metadata["results"][value_name]["unit"]
    else:
        ylabel = ""

    if nice_names:
        df = apply_nice_names(df)

    df = df.reset_index()
    df_melted = df.melt(id_vars=["run", "state"], var_name="result", value_name="value")

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)

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


def get_average_ua2_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get the average UA2 data."""
    df = df.drop(columns=["run"])
    return df.groupby("state").mean().round(1).reset_index()


def _get_ua2_map_color_map(color_palette: str, categories: list[str]) -> dict[str, str]:
    """Get the color map for the UA2 map."""
    try:
        palette = getattr(px.colors.qualitative, color_palette)
    except AttributeError:
        logger.error(f"Color palette {color_palette} not found")
        palette = px.colors.qualitative.Set3
    return {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}


def _get_ua2_map_figure(
    data: list[dict[str, Any]],
    gdf: gpd.GeoDataFrame,
    color_palette: str = "Set3",
    metadata: dict[str, Any] = None,
    **kwargs: Any,
) -> go.Figure:
    """UA2 map component."""

    # to fill in any missing states
    no_data = pd.DataFrame(
        {"state": gdf.STATE_ID.astype(str), "value": np.nan}
    ).set_index("state")

    if not data:
        logger.debug("No map data found")
        df = no_data
    else:
        df = pd.DataFrame(data)
        # get units from metadata
        if metadata:
            value_name = df.columns[-1]
            unit = metadata["results"][value_name]["unit"]
        else:
            unit = ""
        # orientate data for choropleth
        df = df.set_index("state")
        if not len(df.columns) == 1:
            logger.error(f"Expected 1 column, got {len(df.columns)}")
            df = no_data
        else:
            df = df.rename(columns={df.columns[0]: "value"})
            # Convert to numeric for continuous color scale
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = (
        pd.concat([df, no_data], axis=0)
        .reset_index()
        .drop_duplicates(subset=["state"], keep="first")
        .set_index("state")
    )

    # Separate data with values from NaN data
    df_with_data = df[df["value"].notna()].copy()
    df_no_data = df[df["value"].isna()].copy()

    # Create choropleth with only data that has values
    if len(df_with_data) > 0:
        fig = px.choropleth(
            df_with_data,
            geojson=gdf.set_index("STATE_ID"),
            locations=df_with_data.index,
            color="value",
            hover_data=["value"],
            color_continuous_scale=color_palette,
            labels=dict(value=unit),
        )
    else:
        # If no data, create empty figure with just the geojson structure
        fig = px.choropleth(
            df_no_data,
            geojson=gdf.set_index("STATE_ID"),
            locations=df_no_data.index,
            color_continuous_scale=color_palette,
            labels=dict(value=unit),
        )

    # Overlay NaN states in grey
    if len(df_no_data) > 0:
        # Add grey choropleth trace for NaN states using the same geojson
        gdf_indexed = gdf.set_index("STATE_ID")
        # Convert geojson to dict format for go.Choropleth
        geojson_dict = gdf_indexed.__geo_interface__

        nan_trace = go.Choropleth(
            geojson=geojson_dict,
            locations=df_no_data.index,
            z=[1] * len(df_no_data),  # Dummy values, we'll set color to grey
            colorscale=[[0, "lightgrey"], [1, "lightgrey"]],  # All grey
            showscale=False,  # Don't show in legend
            hoverinfo="skip",  # Skip hover for NaN states
            marker_line_width=0,
        )
        fig.add_trace(nan_trace)

    fig.update_geos(
        visible=False,  # remove background world map
        showframe=False,
        showcoastlines=False,
        showland=False,
        showlakes=False,
        showcountries=False,
    )

    # defaults for the hex map
    # overwrite for the actual map
    scale = kwargs.get("scale", 5.9)
    lat = kwargs.get("lat", 44)
    lon = kwargs.get("lon", -100)

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        geo=dict(
            # fitbounds="locations",
            projection=dict(
                # type="albers usa",
                type="mercator",
                scale=scale,
            ),
            center=dict(lat=lat, lon=lon),  # CONUS center
        ),
    )

    return fig


def get_ua2_map(
    map_data: list[dict[str, Any]],
    state_shape: gpd.GeoDataFrame,
    **kwargs: Any,
) -> html.Div:
    """Position maps on a grid system for lazy loading."""

    if not map_data:
        logger.debug("No map data found")
        return html.Div([dbc.Alert("No map data found", color="info")])

    color_scale = kwargs.get("color_scale", DEFAULT_DISCRETE_COLOR_SCALE)
    categories = set(pd.DataFrame(map_data).set_index("state").values.ravel())
    color_map = _get_ua2_map_color_map(color_scale, categories)
    color_map.update({"No Data": "lightgrey"})  # Modify in-place instead of reassigning
    logger.debug(f"Color map: {color_map}")

    # defaults for the hex map
    # overwrite for the actual map
    scale = kwargs.get("scale", 5.9)
    lat = kwargs.get("lat", 44)
    lon = kwargs.get("lon", -100)

    # for extracting ylabels
    metadata = kwargs.get("metadata", {})

    return dcc.Graph(
        id=ids.UA2_MAP,
        figure=_get_ua2_map_figure(
            data=map_data,
            gdf=state_shape,
            color_palette=color_scale,
            color_map=color_map,
            scale=scale,
            lat=lat,
            lon=lon,
            metadata=metadata,
        ),
        # style={"height": "400px"},
    )
