"""Component to display GSA data."""

from typing import Any
from dash import dcc, html, dash_table
from pathlib import Path
from .utils import (
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
    _unflatten_dropdown_options,
)
from . import ids as ids
import dash_bootstrap_components as dbc

import pandas as pd
import geopandas as gpd
import plotly
import plotly.express as px

import logging

logger = logging.getLogger(__name__)

root = Path(__file__).parent.parent

GSA_PARM_OPTIONS = get_gsa_params_dropdown_options(root)
GSA_RESULT_OPTIONS = get_gsa_results_dropdown_options(root)

GSA_RB_OPTIONS = [
    {"label": html.Span("Name", className="ms-2"), "value": "name", "disabled": False},
    {"label": html.Span("Rank", className="ms-2"), "value": "rank", "disabled": False},
]


def _default_gsa_params_value() -> list[str]:
    """Default value for the GSA parameters dropdown."""
    defaults = [
        x["value"]
        for x in GSA_PARM_OPTIONS
        if x["value"].endswith(("_electrical_demand", "_veh_lgt_demand"))
    ]
    if not defaults:
        return GSA_PARM_OPTIONS[0]["value"]
    return defaults


def _default_gsa_results_value() -> list[str]:
    """Default value for the GSA results dropdown."""
    defaults = [
        x["value"]
        for x in GSA_RESULT_OPTIONS
        if any(y in x["value"] for y in ["_energy_", "_carbon"])
    ]
    if not defaults:
        return GSA_RESULT_OPTIONS[0]["value"]
    return defaults


def gsa_options_block() -> html.Div:
    """GSA options block component."""
    return html.Div(
        [
            gsa_filtering_rb(),
            dbc.Collapse(
                [
                    gsa_params_dropdown(),
                ],
                id=ids.GSA_PARAMS_RESULTS_COLLAPSE,
                is_open=True,
            ),
            dbc.Collapse(
                [
                    gsa_params_slider(),
                ],
                id=ids.GSA_PARAMS_SLIDER_COLLAPSE,
                is_open=True,
            ),
            gsa_results_dropdown(),
        ],
    )


def gsa_params_slider() -> html.Div:
    """GSA slider component."""
    max_num = len(GSA_PARM_OPTIONS)
    marks = {
        0: "0",
        max_num // 2: str(max_num // 2),
        max_num: str(max_num),
    }

    return html.Div(
        [
            html.H6("Select top number of parameters:"),
            dcc.Slider(
                id=ids.GSA_PARAMS_SLIDER,
                min=0,
                max=max_num,
                value=5,
                step=1,
                included=False,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
    )


def gsa_filtering_rb() -> html.Div:
    """GSA filtering radio buttons component."""
    return html.Div(
        [
            html.H6(
                "Select parameters by:",
                className="card-title",
            ),
            dcc.RadioItems(
                id=ids.GSA_PARAM_SELECTION_RB,
                options=GSA_RB_OPTIONS,
                value="rank",
                inline=True,
                className="me-3",
                labelStyle={"marginRight": "20px"},
            ),
        ],
        className="dropdown-container",
    )


def gsa_params_dropdown() -> html.Div:
    """GSA parameters dropdown component."""
    return html.Div(
        [
            html.H6("Select Parameter(s)"),
            dcc.Dropdown(
                id=ids.GSA_PARAM_DROPDOWN,
                options=GSA_PARM_OPTIONS,
                value=_default_gsa_params_value(),
                multi=True,
            ),
            html.Div(
                [
                    dbc.Button(
                        "Select All",
                        id=ids.GSA_PARAM_SELECT_ALL,
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                    dbc.Button(
                        "Remove All",
                        id=ids.GSA_PARAM_REMOVE_ALL,
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                ],
            ),
        ],
        className="dropdown-container",
    )


def gsa_results_dropdown() -> html.Div:
    """GSA results dropdown component."""
    return html.Div(
        [
            html.H6("Select Result(s)"),
            dcc.Dropdown(
                id=ids.GSA_RESULTS_DROPDOWN,
                options=GSA_RESULT_OPTIONS,
                value=_default_gsa_results_value(),
                multi=True,
            ),
            html.Div(
                [
                    dbc.Button(
                        "Select All",
                        id=ids.GSA_RESULTS_SELECT_ALL,
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                    dbc.Button(
                        "Remove All",
                        id=ids.GSA_RESULTS_REMOVE_ALL,
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                ]
            ),
        ],
        className="dropdown-container",
    )


def filter_gsa_data(
    data: list[dict[str, Any]] | None,
    param_option: str,
    params_dropdown: str | list[str],
    params_slider: int,
    results: str | list[str],
    keep_iso: bool = False,
) -> pd.DataFrame:
    """Filter GSA data based on selected parameters and results."""
    if not data:
        logger.debug("No raw GSA data available to filter")
        return pd.DataFrame()

    df = pd.DataFrame(data).set_index("param").copy()

    if param_option == "name":
        logger.info("Filtering GSA data by name")
        logger.debug(f"Params: {params_dropdown} | Results: {results}")
        params = params_dropdown
    elif param_option == "rank":
        logger.info("Filtering GSA data by rank")
        logger.debug(f"Num Params: {params_slider} | Results: {results}")
        params = get_top_n_params(df, params_slider, results)
    else:
        logger.debug(f"Invalid flitering GSA selection of: {param_option}")
        return pd.DataFrame()

    if not params or not results:
        logger.debug("No params or results selected for GSA filtering")
        return pd.DataFrame()

    if isinstance(params, str):
        params = [params]
    if isinstance(results, str):
        results = [results]
    if keep_iso:
        logger.debug("Removing ISO in GSA filtering")
        results.append("iso")

    result = df.loc[params][results].copy()

    return result


def filter_gsa_data_for_map(
    data: list[dict[str, Any]] | None,
    params_slider: int,
    result: str,
) -> pd.DataFrame:
    """Filter GSA data for the map."""

    if not data:
        logger.debug("No raw GSA data available to filter")
        return pd.DataFrame()

    if isinstance(result, list):
        logger.debug(f"GSA results not a single result: {result}")
        return pd.DataFrame()

    df = pd.DataFrame(data).set_index(["iso"])[["param", result]]

    df = df.pivot_table(columns="param", index=df.index).T.droplevel(
        0
    )  # level 0 is the result

    for col in df.columns:
        df[col] = df[col].rank(method="dense", ascending=False).astype(int)

    ranking = {}
    for rank in range(1, params_slider):
        ranking[rank] = {}
        for col in df.columns:
            ranking[rank][col] = df[col][df[col] == rank].index[0]

    return pd.DataFrame(ranking)


def get_top_n_params(
    raw: pd.DataFrame, num_params: int, results: list[str]
) -> list[str]:
    """Get the top n most impactful parameters."""

    df = raw.copy()[results]

    if df.empty:
        logger.debug("No top_n parameters found for GSA")
        return []

    top_n = []
    for col in df.columns:
        top_n.extend(df[col].sort_values(ascending=False).index[:num_params].to_list())
    return sorted(list(set(top_n)))


def _get_heatmap_height(num_params: int) -> int:
    """Get the height of the heatmap based on the number of parameters."""
    height = num_params * 20
    if height < 800:
        return 800
    return height


def get_gsa_heatmap(
    data: dict[str, Any], nice_names: bool = True, **kwargs: Any
) -> plotly.graph_objects.Figure:
    """GSA heatmap component."""

    if not data:
        logger.debug("No GSA heatmap data found")
        return px.imshow(
            pd.DataFrame(),
            color_continuous_scale="Bluered",
            color_continuous_midpoint=0,
            zmin=0,
            zmax=1,
            aspect="auto",
        )

    df = pd.DataFrame(data).set_index("param")
    logger.debug(f"Heatmap GSA data shape: {df.shape}")

    if nice_names:
        logger.debug("Applying nice names to GSA heatmap")
        gsa_params = _unflatten_dropdown_options(GSA_PARM_OPTIONS)
        gsa_results = _unflatten_dropdown_options(GSA_RESULT_OPTIONS)
        df = df.rename(columns=gsa_results).rename(index=gsa_params)

    color_scale = kwargs.get("color_scale", "PuBu")
    logger.debug(f"GSA heatmap color scale: {color_scale}")

    fig = px.imshow(
        df,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=0,
        zmin=0,
        zmax=1,
        aspect="auto",
        labels=dict(x="Parameters", y="Results", color="Scaled Elementary Effect"),
    )

    fig.update_layout(
        title="Global Sensitivity Analysis",
        xaxis_title="",
        yaxis_title="",
        height=_get_heatmap_height(len(df.index)),
        xaxis={"side": "bottom"},
    )
    fig.update_xaxes(tickangle=45)

    return fig


def get_gsa_data_table(
    data: dict[str, Any], nice_names: bool = True
) -> dash_table.DataTable:
    """GSA data table component."""
    if not data:
        logger.debug("No GSA data table data found")
        return dash_table.DataTable(
            data=[],
            columns=[],
            style_table={"overflowX": "auto"},
        )

    df = pd.DataFrame(data).set_index("param")
    logger.debug(f"GSA data table shape: {df.shape}")

    if nice_names:
        logger.debug("Applying nice names to GSA data table")
        gsa_params = _unflatten_dropdown_options(GSA_PARM_OPTIONS)
        gsa_results = _unflatten_dropdown_options(GSA_RESULT_OPTIONS)
        df = df.rename(columns=gsa_results).rename(index=gsa_params)

    df = df.reset_index()

    # Format columns for display
    columns = [
        {"name": "Parameter", "id": "param", "type": "text"},
    ] + [
        {"name": col, "id": col, "type": "numeric", "format": {"specifier": ".3f"}}
        for col in df.columns
        if col != "param"
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


def normalize_mu_star_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the data to be between 0 and 1."""
    if data.empty:
        logger.debug("No GSA data to normalize")
        return pd.DataFrame()

    df = data.copy().set_index("param")

    for column in df.columns:
        if column == "iso":
            continue
        max_value = df[column].max()
        df[column] = df[column].div(max_value)

    return df.reset_index()


def get_gsa_barchart(
    normed_data: dict[str, Any], nice_names: bool = True
) -> plotly.graph_objects.Figure:
    """GSA barchart component."""
    if not normed_data:
        logger.debug("No nomred GSA data found")
        return px.bar(pd.DataFrame(), x="param", y="value", color="param")

    df = pd.DataFrame(normed_data).set_index("param")

    logger.debug(f"GSA barchart data shape: {df.shape}")

    if nice_names:
        logger.debug("Applying nice names to GSA barchart")
        gsa_params = _unflatten_dropdown_options(GSA_PARM_OPTIONS)
        gsa_results = _unflatten_dropdown_options(GSA_RESULT_OPTIONS)
        df = df.rename(columns=gsa_results).rename(index=gsa_params)

    # melt to get it in long format for plotting
    df_melted = df.reset_index().melt(
        id_vars=["param"], var_name="result", value_name="value"
    )

    fig = px.bar(
        df_melted,
        x="value",
        y="param",
        color="result",
        orientation="h",
        barmode="group",
        labels={"value": "µ*/µ*_max", "param": "Parameter", "result": "Result"},
        title="Scaled µ*",
    )

    fig.update_layout(
        height=max(800, len(df.index) * 100),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis=dict(range=[0, 1], title_standoff=10),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def get_gsa_map(
    data: list[dict[str, Any]], gdf: gpd.GeoDataFrame, top_n: int = 1
) -> plotly.graph_objects.Figure:
    """GSA map component."""

    # to fill in any missing ISOs
    no_data = pd.DataFrame({"iso": gdf.iso.astype(str), "value": "No Data"}).set_index(
        "iso"
    )

    if not data:
        logger.debug("No GSA map data found")
        rankings = no_data
    else:
        rankings = pd.DataFrame(data, dtype=str)
        rankings = rankings.set_index("iso").astype(str)
        rankings = rankings.iloc[:, top_n - 1].rename("value")

    rankings = (
        no_data.join(rankings, how="left", lsuffix="_drop")
        .fillna("No Data")
        .drop(columns=["value_drop"])
    )

    categories = rankings["value"].unique()
    color_map = {
        cat: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
        for i, cat in enumerate(categories)
    }

    logger.debug(f"Rankings:\n{rankings}")
    logger.debug(f"GDF:\n{gdf}")

    fig = px.choropleth(
        rankings,
        geojson=gdf.set_index("iso"),
        locations=rankings.index,
        color="value",
        hover_data=["value"],
        color_discrete_map=color_map,
    )

    fig.update_geos(
        visible=False,  # remove background world map
        showframe=False,
        showcoastlines=False,
        showland=False,
        showlakes=False,
        showcountries=False,
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        geo=dict(
            fitbounds="locations",  # fit the map to geometries
            projection=dict(type="albers usa"),
            scope="usa",
        ),
    )

    return fig
