from pathlib import Path
from typing import Any
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

from components.utils import (
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
)
import components.ids as ids
from components.gsa import (
    filter_gsa_data,
    gsa_options_block,
    get_gsa_heatmap,
    get_top_n_params,
    get_gsa_data_table,
    get_gsa_barchart,
    normalize_mu_star_data,
)
from components.shared import iso_options_block, plotting_options_block
from components.ua import ua_options_block

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

RAW_GSA = pd.read_csv("data/sa.csv")
RAW_UA = pd.read_csv("data/ua_runs.csv")
ISO_SHAPE = gpd.read_file("data/iso_shapes.geojson")

root = Path(__file__).parent
GSA_PARM_OPTIONS = get_gsa_params_dropdown_options(root)
GSA_RESULT_OPTIONS = get_gsa_results_dropdown_options(root)

###
# Define the layout
###

app.layout = html.Div(
    [
        dbc.NavbarSimple(
            brand="High Impact Options to Reach Near Term Targets",
            brand_href="#",
            color="primary",
            dark=True,
            id=ids.NAVBAR,
        ),
        dbc.Container(
            [
                html.H1("PyPSA-USA Uncertainity Analysis", className="my-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Data Viewing Options",
                                                    className="card-title",
                                                ),
                                                plotting_options_block(),
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Collapse(
                                                    [
                                                        html.H4(
                                                            "Spatial Options",
                                                            className="card-title",
                                                        ),
                                                        iso_options_block(
                                                            RAW_GSA, RAW_UA
                                                        ),
                                                    ],
                                                    id=ids.ISO_OPTIONS_BLOCK,
                                                    is_open=True,
                                                ),
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Collapse(
                                                    [
                                                        html.H4(
                                                            "GSA Options",
                                                            className="card-title",
                                                        ),
                                                        gsa_options_block(),
                                                    ],
                                                    id=ids.GSA_OPTIONS_BLOCK,
                                                    is_open=True,
                                                ),
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Collapse(
                                                    [
                                                        html.H4(
                                                            "Uncertaintiy Options",
                                                            className="card-title",
                                                        ),
                                                        ua_options_block(),
                                                    ],
                                                    id=ids.UA_OPTIONS_BLOCK,
                                                    is_open=True,
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            md=2,  # Takes up 1/6 of the width
                        ),
                        dbc.Col(
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            label="Input Data",
                                            tab_id=ids.DATA_TAB,
                                        ),
                                        dbc.Tab(
                                            label="Sensitivity Analysis",
                                            tab_id=ids.SA_TAB,
                                        ),
                                        dbc.Tab(
                                            label="Uncertainty Analysis",
                                            tab_id=ids.UA_TAB,
                                        ),
                                    ],
                                    id=ids.TABS,
                                    active_tab=ids.SA_TAB,
                                    className="mb-4",
                                ),
                                html.Div(id=ids.TAB_CONTENT),
                            ],
                            md=10,
                        ),
                    ]
                ),
                # Store components to share data between callbacks
                dcc.Store(id=ids.SA_STORE),  # raw data
                dcc.Store(id=ids.UA_STORE),
                dcc.Store(id=ids.SA_DATA_TABLE),  # filtered by iso
                dcc.Store(id=ids.UA_DATA_TABLE),
                dcc.Store(id=ids.GSA_NORMED), # mu/mu_max calculation
                dcc.Store(id=ids.GSA_NORMED_DATA_TABLE), 
                dcc.Store(id=ids.GSA_PARAM_BUTTON_STATE, data=""),
                dcc.Store(id=ids.GSA_RESULTS_BUTTON_STATE, data=""),
            ],
            fluid=True,
        ),
    ]
)

###
# Update tab content
###


@app.callback(
    Output(ids.TAB_CONTENT, "children"),
    [
        Input(ids.TABS, "active_tab"),
        Input(ids.SA_DATA_TABLE, "data"),
        Input(ids.GSA_NORMED_DATA_TABLE, "data"),
        Input(ids.UA_STORE, "data"),
        Input(ids.PLOTTING_TYPE_DROPDOWN, "value"),
        Input(ids.COLOR_SCALE_DROPDOWN, "value"),
    ],
)
def render_tab_content(
    active_tab, sa_data, gsa_normed_data, ua_data, plotting_type, color_scale
) -> html.Div:
    logger.debug(f"Active Tab: {active_tab}")
    if active_tab == ids.DATA_TAB:
        return html.Div()
    elif active_tab == ids.SA_TAB:
        if plotting_type == "heatmap":
            view = dcc.Graph(
                id=ids.GSA_HEATMAP,
                figure=get_gsa_heatmap(sa_data, color_scale=color_scale),
            )
        elif plotting_type == "data_table":
            view = get_gsa_data_table(sa_data)
        elif plotting_type == "barchart":
            view = dcc.Graph(
                id=ids.GSA_BAR_CHART,
                figure=get_gsa_barchart(gsa_normed_data),
            )
        elif plotting_type == "map":
            view = dcc.Graph(
                id=ids.GSA_MAP,
                figure=get_gsa_map(sa_data),
            )

        return html.Div([dbc.Card([dbc.CardBody([view])])])
    elif active_tab == ids.UA_TAB:
        return html.Div()
    else:
        return html.Div([dbc.Alert("No data found", color="info")])


@app.callback(
    Output(ids.PLOTTING_TYPE_DROPDOWN, "options"),
    Input(ids.TABS, "active_tab"),
)
def update_plotting_type_dropdown_options(active_tab: str) -> list[dict[str, str]]:
    """Update the plotting type dropdown options based on the active tab."""
    logger.debug(f"Active Tab: {active_tab}")
    if active_tab == ids.DATA_TAB:
        return [{"label": "Data Table", "value": "data_table"}]
    elif active_tab == ids.SA_TAB:
        return [
            {"label": "Data Table", "value": "data_table"},
            {"label": "Heatmap", "value": "heatmap"},
            {"label": "Bar Chart", "value": "barchart"},
            {"label": "Map", "value": "map"},
        ]
    else:
        logger.debug(f"Invalid active tab: {active_tab}")
        return [{"label": "Data Table", "value": "data_table"}]


###
# Get stored data
###

# callbacks to filter raw data based on ISO since ISO dropdown shared between GSA and UA


@app.callback(
    Output(ids.SA_STORE, "data"),
    Input(ids.ISO_DROPDOWN, "value"),
)
def filter_raw_data_gsa(isos: list[str]) -> dict[str, Any]:
    """Update the GSA store data based on the selected ISOs."""
    logger.debug(f"ISO dropdown value: {isos}")
    if not isos:
        logger.debug("No ISOs selected")
        return None  # Change empty list to None to be more explicit
    data = RAW_GSA[RAW_GSA.iso.isin(isos)].drop(columns=["iso"]).to_dict("records")
    logger.debug(f"SA_STORE data length: {len(data)}")
    return data


@app.callback(
    Output(ids.GSA_NORMED, "data"),
    Input(ids.SA_STORE, "data"),
)
def normalize_mu_star(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize data for the mu/mu_max calculation.

    Treat this as a store so that it is not recalculated every time the data is filtered,
    as this is independent of the result selections.
    """
    df = pd.DataFrame(data)
    return normalize_mu_star_data(df).to_dict("records")

# Original callback for data table
@app.callback(
    Output(ids.SA_DATA_TABLE, "data"),
    [
        Input(ids.GSA_PARAM_SELECTION_RB, "value"),
        Input(ids.GSA_PARAM_DROPDOWN, "value"),
        Input(ids.GSA_PARAMS_SLIDER, "value"),
        Input(ids.GSA_RESULTS_DROPDOWN, "value"),
        Input(ids.SA_STORE, "data"),
    ],
)
def update_gsa_si_data(
    param_option: str,
    params_dropdown: str | list[str],
    params_slider: int,
    results: str | list[str],
    raw_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Update the GSA SI data for the data table."""
    df = filter_gsa_data(param_option, params_dropdown, params_slider, results, raw_data)
    return df.reset_index().to_dict("records")


# New callback for barchart
@app.callback(
    Output(ids.GSA_NORMED_DATA_TABLE, "data"),
    [
        Input(ids.GSA_PARAM_SELECTION_RB, "value"),
        Input(ids.GSA_PARAM_DROPDOWN, "value"),
        Input(ids.GSA_PARAMS_SLIDER, "value"),
        Input(ids.GSA_RESULTS_DROPDOWN, "value"),
        Input(ids.GSA_NORMED, "data"),
    ],
)
def update_gsa_si_nomred_data(
    param_option: str,
    params_dropdown: str | list[str],
    params_slider: int,
    results: str | list[str],
    normed_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Update the GSA barchart."""
    df = filter_gsa_data(param_option, params_dropdown, params_slider, results, normed_data)
    return df.reset_index().to_dict("records")


### UA


@app.callback(
    Output(ids.UA_STORE, "data"),
    Input(ids.ISO_DROPDOWN, "value"),
)
def filter_raw_data_ua(isos: list[str]) -> dict[str, Any]:
    return RAW_GSA[RAW_GSA.iso.isin(isos)].drop(columns=["iso"]).to_dict("records")


###
# Enable/disable collapsable blocks
###


@app.callback(
    [
        Output(ids.ISO_OPTIONS_BLOCK, "is_open"),
        Output(ids.GSA_OPTIONS_BLOCK, "is_open"),
        Output(ids.UA_OPTIONS_BLOCK, "is_open"),
    ],
    Input(ids.TABS, "active_tab"),
)
def enable_disable_option_blocks(active_tab: str) -> tuple[bool, bool, bool]:
    """Enable/disable dropdowns based on active tab and filtering mode."""
    # Start with all disabled
    iso_open = False
    gsa_open = False
    ua_open = False
    if active_tab == ids.SA_TAB:
        iso_open = True
        gsa_open = True
    elif active_tab == ids.UA_TAB:
        iso_open = True
        gsa_open = True

    return iso_open, gsa_open, ua_open


@app.callback(
    [
        Output(ids.GSA_PARAMS_SLIDER_COLLAPSE, "is_open"),
        Output(ids.GSA_PARAMS_RESULTS_COLLAPSE, "is_open"),
    ],
    Input(ids.GSA_PARAM_SELECTION_RB, "value"),
)
def enable_disable_gsa_param_selection(value: str) -> tuple[bool, bool]:
    """Enable/disable GSA parameter selection collapse based on filtering mode."""
    params_slider_open = False
    params_dropdown_open = False
    if value == "rank":
        params_slider_open = True
    elif value == "name":
        params_dropdown_open = True
    else:
        logger.debug(f"Invalid value for GSA parameter selection: {value}")
        params_slider_open = True
        params_dropdown_open = True
    return params_slider_open, params_dropdown_open


###
# GSA Options Callbacks
###


@app.callback(
    Output(ids.GSA_PARAM_DROPDOWN, "value"),
    [
        Input(ids.GSA_PARAM_SELECT_ALL, "n_clicks"),
        Input(ids.GSA_PARAM_REMOVE_ALL, "n_clicks"),
    ],
    prevent_initial_call=True,
)
def select_gsa_params(*args: Any) -> list[str]:
    """Select/remove all parameters when button is clicked."""
    ctx = dash.callback_context  # to determine which button was clicked
    if not ctx.triggered:
        return dash.no_update

    # id of the button that was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    logger.debug(f"Button ID: {button_id}")

    if button_id == ids.GSA_PARAM_SELECT_ALL:
        return [option["value"] for option in GSA_PARM_OPTIONS]
    elif button_id == ids.GSA_PARAM_REMOVE_ALL:
        return []
    return dash.no_update


@app.callback(
    Output(ids.GSA_RESULTS_DROPDOWN, "value"),
    [
        Input(ids.GSA_RESULTS_SELECT_ALL, "n_clicks"),
        Input(ids.GSA_RESULTS_REMOVE_ALL, "n_clicks"),
    ],
    prevent_initial_call=True,
)
def select_gsa_results(*args: Any) -> list[str]:
    """Select/remove all results when button is clicked."""
    ctx = dash.callback_context  # to determine which button was clicked
    if not ctx.triggered:
        return dash.no_update

    # id of the button that was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    logger.debug(f"Button ID: {button_id}")

    if button_id == ids.GSA_RESULTS_SELECT_ALL:
        return [option["value"] for option in GSA_RESULT_OPTIONS]
    elif button_id == ids.GSA_RESULTS_REMOVE_ALL:
        return []
    return dash.no_update


# Run the server
if __name__ == "__main__":
    app.run(debug=True)
