"""Component to display GSA data."""

from typing import Any
from dash import dcc, html, dash_table
from pathlib import Path
from .utils import (
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
    _unflatten_dropdown_options,
    DEFAULT_CONTINOUS_COLOR_SCALE,
    DEFAULT_DISCRETE_COLOR_SCALE,
)
from .data import GSA_RESULT_OPTIONS, GSA_PARM_OPTIONS
from . import ids as ids
from .styles import BUTTON_STYLE, DATA_TABLE_STYLE
import dash_bootstrap_components as dbc

import pandas as pd
import geopandas as gpd
import plotly
import plotly.express as px

import logging

logger = logging.getLogger(__name__)

GSA_RB_OPTIONS = [
    {"label": html.Span("Name", className="ms-2"), "value": "name", "disabled": False},
    {"label": html.Span("Rank", className="ms-2"), "value": "rank", "disabled": False},
]


def _default_gsa_params_value(options: list[dict[str, str]]) -> list[str]:
    """Default value for the GSA parameters dropdown."""
    defaults = [
        x["value"]
        for x in options
        if x["value"].endswith(("_electrical_demand", "_veh_lgt_demand"))
    ]
    if not defaults:
        return options[0]["value"]
    return defaults


def _default_gsa_results_value() -> list[str]:
    """Default value for the GSA results dropdown."""
    defaults = [
        x["value"]
        for x in GSA_RESULT_OPTIONS
        if any(y in x["value"] for y in ["_energy_", "carbon_"])
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
                value=_default_gsa_params_value(GSA_PARM_OPTIONS),
                multi=True,
            ),
            html.Div(
                [
                    dbc.Button(
                        "Select All",
                        id=ids.GSA_PARAM_SELECT_ALL,
                        **BUTTON_STYLE,
                    ),
                    dbc.Button(
                        "Remove All",
                        id=ids.GSA_PARAM_REMOVE_ALL,
                        **BUTTON_STYLE,
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
                        **BUTTON_STYLE,
                    ),
                    dbc.Button(
                        "Remove All",
                        id=ids.GSA_RESULTS_REMOVE_ALL,
                        **BUTTON_STYLE,
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

    if isinstance(params_dropdown, str):
        params_dropdown = [params_dropdown]
    if isinstance(results, str):
        results = [results]

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
    nice_names: bool = True,
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
    for rank in range(1, params_slider + 1):  # rankings start at 1, not zero!
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
    if height < 700:
        return 700
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

    color_scale = kwargs.get("color_scale", DEFAULT_CONTINOUS_COLOR_SCALE)
    logger.debug(f"GSA heatmap color scale: {color_scale}")

    fig = px.imshow(
        df,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=0,
        zmin=0,
        # zmax=1,
        aspect="auto",
        labels=dict(x="Parameters", y="Results", color="Scaled Elementary Effect"),
    )

    fig.update_layout(
        title="",
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
        **DATA_TABLE_STYLE,
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
    normed_data: dict[str, Any],
    nice_names: bool = True,
    **kwargs: Any,
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

    color_scale = kwargs.get("color_scale", DEFAULT_DISCRETE_COLOR_SCALE)

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
        title="",
        height=max(800, len(df.index) * 100),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis=dict(range=[0, 1], title_standoff=10),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def _get_gsa_map_color_map(color_palette: str, categories: list[str]) -> dict[str, str]:
    """Get the color map for the GSA map."""
    palette = getattr(px.colors.qualitative, color_palette)
    return {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}


def _get_gsa_map_figure(
    data: list[dict[str, Any]],
    gdf: gpd.GeoDataFrame,
    top_n: int = 0,
    color_map: dict[str, str] | None = None,
    color_palette: str = "Set3",
) -> plotly.graph_objects.Figure:
    """GSA map component."""

    if top_n <= 0:
        logger.debug(f"Top n is {top_n}, setting to 1")
        top_n = 1

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
        rankings = rankings.loc[:, str(top_n)].rename("value")

    rankings = (
        no_data.join(rankings, how="left", lsuffix="_drop")
        .fillna("No Data")
        .drop(columns=["value_drop"])
    )

    categories = rankings["value"].unique()

    if color_map is None:
        color_map = _get_gsa_map_color_map(color_palette, categories)

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
            # fitbounds="locations",
            projection=dict(
                type="albers usa",
                scale=0.85,
            ),
            center=dict(lat=39, lon=-95),  # CONUS center
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            title=None,
        ),
    )

    return fig


def get_gsa_map(
    gsa_map_data: list[dict[str, Any]],
    iso_shape: gpd.GeoDataFrame,
    num_cols: int = 2,
    card_class: str = "h-100",
    row_class: str = "mb-2 g-2",  # Adds margin bottom and gap between cards
    **kwargs: Any,
) -> html.Div:
    """Position maps on a grid system for lazy loading."""

    if not gsa_map_data:
        num_maps = 1  # print an empty map
    else:
        num_maps = len(gsa_map_data[0]) - 1  # minus 1 as iso is included
        logger.debug(f"User input for top params of: {num_maps}")

    color_scale = kwargs.get("color_scale", DEFAULT_DISCRETE_COLOR_SCALE)
    categories = set(pd.DataFrame(gsa_map_data).set_index("iso").values.ravel())
    color_map = _get_gsa_map_color_map(color_scale, categories)
    color_map.update({"No Data": "lightgrey"})  # Modify in-place instead of reassigning
    logger.debug(f"Color map: {color_map}")

    if num_maps == 1:
        return dcc.Graph(
            id=ids.GSA_MAP,
            figure=_get_gsa_map_figure(
                data=gsa_map_data,
                gdf=iso_shape,
                top_n=num_maps,
                color_palette=color_scale,
                color_map=color_map,
            ),
            style={"height": "400px"},
        )

    # bootstrap uses 12 columns
    col_width = int(12 / num_cols)

    logger.debug(f"Creating num of maps: {num_maps}")

    cards = []
    for num_map in range(1, num_maps + 1):  # indixing of rank starts at 1
        logger.debug(f"Creating card for map {num_map}")
        card = dbc.Card(
            [
                dbc.CardHeader(f"No. {num_map} Parameter"),
                dbc.CardBody(
                    [
                        dcc.Graph(
                            id=f"{ids.GSA_MAP}-{num_map}",
                            figure=_get_gsa_map_figure(
                                data=gsa_map_data,
                                gdf=iso_shape,
                                top_n=num_map,
                                # color_palette=color_scale,
                                color_map=color_map,
                            ),
                            style={"height": "300px"},  # fixed height for map
                        )
                    ]
                ),
            ],
            className=card_class,
        )
        cards.append(card)

    rows = []
    for i in range(0, len(cards), num_cols):
        row_cards = cards[i : i + num_cols]
        cols = [dbc.Col(card, width=col_width) for card in row_cards]
        rows.append(dbc.Row(cols, className=row_class))

    return html.Div(rows, id=ids.GSA_MAP)
