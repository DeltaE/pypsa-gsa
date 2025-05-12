import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import components.ids as ids
import logging

from components.gsa import gsa_params_dropdown, gsa_results_dropdown
from components.shared import iso_dropdown
from components.ua import ua_params_dropdown, ua_results_dropdown

logger = logging.getLogger(__name__)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

# Define the layout
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
                                                iso_dropdown(),
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
                dcc.Store(id=ids.SA_STORE),
                dcc.Store(id=ids.UA_STORE),
            ],
            fluid=True,
        ),
    ]
)


# Callback to update tab content
@app.callback(
    Output(ids.TAB_CONTENT, "children"),
    [
        Input(ids.TABS, "active_tab"),
        Input(ids.SA_STORE, "data"),
        Input(ids.UA_STORE, "data"),
    ],
)
def render_tab_content(active_tab, sa_data, ua_data) -> html.Div:
    if sa_data is None:
        return html.Div()
    if ua_data is None:
        return html.Div()

    if active_tab == ids.DATA_TAB:
        return html.Div()
    elif active_tab == ids.SA_TAB:
        return html.Div()
    elif active_tab == ids.UA_TAB:
        return html.Div()
    else:
        return html.Div([dbc.Alert("No data found", color="info")])

# callback for enable/disable dropdowns
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


# Run the server
if __name__ == "__main__":
    app.run(debug=True)
