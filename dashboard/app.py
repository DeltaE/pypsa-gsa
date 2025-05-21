from pathlib import Path
from typing import Any
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.utils import get_gsa_params_dropdown_options, get_gsa_results_dropdown_options, get_iso_dropdown_options
import components.ids as ids
from components.gsa import gsa_params_dropdown, gsa_results_dropdown, get_gsa_heatmap
from components.shared import iso_dropdown
from components.ua import ua_params_dropdown, ua_results_dropdown

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

RAW_GSA = pd.read_csv("data/sa.csv")
RAW_UA = pd.read_csv("data/ua_runs.csv")

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
                                                    "Spatial Options",
                                                    className="card-title",
                                                ),
                                                iso_dropdown(RAW_GSA, RAW_UA),
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "GSA Options",
                                                    className="card-title",
                                                ),
                                                gsa_params_dropdown(),
                                                gsa_results_dropdown(),
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Uncertaintiy Options",
                                                    className="card-title",
                                                ),
                                                ua_params_dropdown(),
                                                ua_results_dropdown(),
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
                dcc.Store(id=ids.SA_STORE), # raw data
                dcc.Store(id=ids.UA_STORE),
                dcc.Store(id=ids.SA_DATA_TABLE), # filtered by iso
                dcc.Store(id=ids.UA_DATA_TABLE),
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
    ],
    [
        State(ids.UA_STORE, "data"),
    ],
)
def render_tab_content(active_tab, sa_data, ua_data) -> html.Div:
    logger.debug(f"Active Tab: {active_tab}")
    if active_tab == ids.DATA_TAB:
        return html.Div()
    elif active_tab == ids.SA_TAB:
        return html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id=ids.GSA_HEATMAP, figure=get_gsa_heatmap(sa_data)
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    elif active_tab == ids.UA_TAB:
        return html.Div()
    else:
        return html.Div([dbc.Alert("No data found", color="info")])

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

# callback for updating of GSA data based on parameters and results
@app.callback(
    Output(ids.SA_DATA_TABLE, "data"),
    [
        Input(ids.GSA_PARAM_DROPDOWN, "value"),
        Input(ids.GSA_RESULTS_DROPDOWN, "value"),
        Input(ids.SA_STORE, "data"), 
    ],
)
def update_gsa_si_data(
    params: str | list[str], results: str | list[str], raw: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Update the GSA SI data based on the selected parameters and results."""
    logger.debug(f"Params: {params}, Results: {results}")
    logger.debug(f"Raw data type: {type(raw)}")

    if not raw:
        logger.debug("No raw data available")
        return []

    if not params or not results:
        logger.debug("No params or results selected")
        return []

    if isinstance(params, str):
        params = [params]
    if isinstance(results, str):
        results = [results]
    
    raw_df = pd.DataFrame(raw).set_index("param").copy()
    
    logger.debug(f"Length raw dataframe columns (results): {len(raw_df.columns)}")
    logger.debug(f"Length raw dataframe index (params): {len(raw_df.index)}")
    
    result = raw_df.loc[params][results].copy()
    
    logger.debug(f"Length filtered dataframe columns (results): {len(result.columns)}")
    logger.debug(f"Length filtered dataframe index (params): {len(result.index)}")
    
    return result.reset_index().to_dict("records")


### UA

@app.callback(
    Output(ids.UA_STORE, "data"),
    Input(ids.ISO_DROPDOWN, "value"),
)
def filter_raw_data_gsa(isos: list[str]) -> dict[str, Any]:
    return RAW_GSA[RAW_GSA.iso.isin(isos)].drop(columns=["iso"]).to_dict("records")


###
# Enable/disable dropdowns
###

@app.callback(
    [
        Output(ids.ISO_DROPDOWN, "disabled"),
        Output(ids.GSA_PARAM_DROPDOWN, "disabled"),
        Output(ids.GSA_RESULTS_DROPDOWN, "disabled"),
        Output(ids.UA_PARAM_DROPDOWN, "disabled"),
        Output(ids.UA_RESULTS_DROPDOWN, "disabled"),
    ],
    Input(ids.TABS, "active_tab"),
)
def enable_disable_dropdowns(active_tab):
    isos = True
    gsa_params = True
    gsa_results = True
    ua_params = True
    ua_results = True
    if active_tab == ids.SA_TAB:
        isos = False
        gsa_params = False
        gsa_results = False
    elif active_tab == ids.UA_TAB:
        isos = False
        ua_params = False
        ua_results = False
    return isos, gsa_params, gsa_results, ua_params, ua_results

###
# GSA Button Callbacks
###

@app.callback(
    Output(ids.GSA_PARAM_DROPDOWN, "value"),
    [
        Input(ids.GSA_PARAM_SELECT_ALL, "n_clicks"),
        Input(ids.GSA_PARAM_REMOVE_ALL, "n_clicks"),
    ],
    prevent_initial_call=True,
)
def select_gsa_params(*args):
    """Select/remove all parameters when button is clicked."""
    ctx = dash.callback_context # to determine which button was clicked
    if not ctx.triggered:
        return dash.no_update
    
    # id of the button that was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0] 
    
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
def select_gsa_results(*args):
    """Select/remove all results when button is clicked."""
    ctx = dash.callback_context # to determine which button was clicked
    if not ctx.triggered:
        return dash.no_update
    
    # id of the button that was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == ids.GSA_RESULTS_SELECT_ALL:
        return [option["value"] for option in GSA_RESULT_OPTIONS]
    elif button_id == ids.GSA_RESULTS_REMOVE_ALL:
        return []
    return dash.no_update

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
