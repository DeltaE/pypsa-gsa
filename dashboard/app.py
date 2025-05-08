import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
import io

from components.ids import SA_TAB, UP_TAB

# Initialize the Dash app with Bootstrap styling

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
        ),
        dbc.Container(
            [
                html.H1("Uncertainity Analysis Toolbar", className="my-4"),
                # Page selection tabs
                dbc.Tabs(
                    [
                        dbc.Tab(label="Sensitivity Analysis", tab_id=SA_TAB),
                        dbc.Tab(
                            label="Uncertainty Propagation",
                            tab_id=UP_TAB,
                        ),
                    ],
                    id="tabs",
                    active_tab=SA_TAB,
                    className="mb-4",
                ),
                # File upload component (common to both tabs)
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Upload Data"),
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select CSV File")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="output-data-upload"),
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                # Content will change based on selected tab
                html.Div(id="tab-content"),
                # Store components to share data between callbacks
                dcc.Store(id="stored-data"),
                dcc.Store(id="selected-columns"),
            ],
            fluid=True,
        ),
    ]
)


# Callback to parse uploaded data
@app.callback(
    [Output("stored-data", "data"), Output("output-data-upload", "children")],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")],
)
def update_output(contents, filename):
    if contents is None:
        return None, None

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        if "csv" in filename:
            # Read the CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

            # Return the data as JSON for storage and display a preview table
            return df.to_json(date_format="iso", orient="split"), [
                html.Div(
                    [
                        html.H5(f"File: {filename}"),
                        html.H6("Data Preview:"),
                        dbc.Table.from_dataframe(
                            df.head(),
                            striped=True,
                            bordered=True,
                            hover=True,
                            responsive=True,
                            size="sm",
                        ),
                    ]
                )
            ]
    except Exception as e:
        return None, [dbc.Alert(f"Error processing file: {e}", color="danger")]


# Callback to update tab content
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("stored-data", "data")],
)
def render_tab_content(active_tab, stored_data):
    if stored_data is None:
        return html.Div(
            [dbc.Alert("Please upload a CSV file to begin analysis", color="info")]
        )

    # Parse the stored data back to a DataFrame
    df = pd.read_json(stored_data, orient="split")
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    all_columns = df.columns.tolist()

    if active_tab == "sensitivity-analysis":
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Sensitivity Analysis Configuration"
                                                ),
                                                html.Hr(),
                                                # Variable selection
                                                dbc.FormGroup(
                                                    [
                                                        dbc.Label("Select X Variable"),
                                                        dcc.Dropdown(
                                                            id="x-variable",
                                                            options=[
                                                                {
                                                                    "label": col,
                                                                    "value": col,
                                                                }
                                                                for col in numeric_columns
                                                            ],
                                                            value=(
                                                                numeric_columns[0]
                                                                if numeric_columns
                                                                else None
                                                            ),
                                                        ),
                                                    ]
                                                ),
                                                dbc.FormGroup(
                                                    [
                                                        dbc.Label("Select Y Variable"),
                                                        dcc.Dropdown(
                                                            id="y-variable",
                                                            options=[
                                                                {
                                                                    "label": col,
                                                                    "value": col,
                                                                }
                                                                for col in numeric_columns
                                                            ],
                                                            value=(
                                                                numeric_columns[1]
                                                                if len(numeric_columns)
                                                                > 1
                                                                else (
                                                                    numeric_columns[0]
                                                                    if numeric_columns
                                                                    else None
                                                                )
                                                            ),
                                                        ),
                                                    ]
                                                ),
                                                # Filter options
                                                dbc.FormGroup(
                                                    [
                                                        dbc.Label("Filter by Column"),
                                                        dcc.Dropdown(
                                                            id="filter-column",
                                                            options=[
                                                                {
                                                                    "label": "None",
                                                                    "value": "None",
                                                                }
                                                            ]
                                                            + [
                                                                {
                                                                    "label": col,
                                                                    "value": col,
                                                                }
                                                                for col in all_columns
                                                            ],
                                                            value="None",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(id="filter-value-container"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Sensitivity Analysis Results"),
                                                dcc.Graph(id="sensitivity-plot"),
                                                html.Div(id="sensitivity-stats"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            md=8,
                        ),
                    ]
                )
            ]
        )

    elif active_tab == "uncertainty-propagation":
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Uncertainty Analysis Configuration"
                                                ),
                                                html.Hr(),
                                                # Parameter selection
                                                dbc.FormGroup(
                                                    [
                                                        dbc.Label(
                                                            "Select Parameter to Analyze"
                                                        ),
                                                        dcc.Dropdown(
                                                            id="parameter-select",
                                                            options=[
                                                                {
                                                                    "label": col,
                                                                    "value": col,
                                                                }
                                                                for col in numeric_columns
                                                            ],
                                                            value=(
                                                                numeric_columns[0]
                                                                if numeric_columns
                                                                else None
                                                            ),
                                                        ),
                                                    ]
                                                ),
                                                # Grouping options
                                                dbc.FormGroup(
                                                    [
                                                        dbc.Label("Group by Category"),
                                                        dcc.Dropdown(
                                                            id="group-column",
                                                            options=[
                                                                {
                                                                    "label": "None",
                                                                    "value": "None",
                                                                }
                                                            ]
                                                            + [
                                                                {
                                                                    "label": col,
                                                                    "value": col,
                                                                }
                                                                for col in all_columns
                                                            ],
                                                            value="None",
                                                        ),
                                                    ]
                                                ),
                                                html.Div(id="group-value-container"),
                                                # Monte Carlo simulation options (only shown when no grouping)
                                                html.Div(id="monte-carlo-options"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Uncertainty Analysis Results"),
                                                dcc.Graph(id="uncertainty-plot"),
                                                html.Div(id="uncertainty-stats"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            md=8,
                        ),
                    ]
                )
            ]
        )


# Callback to update filter value dropdown based on selected column
@app.callback(
    Output("filter-value-container", "children"),
    [Input("filter-column", "value")],
    [State("stored-data", "data")],
)
def update_filter_value_dropdown(filter_column, stored_data):
    if filter_column == "None" or stored_data is None:
        return []

    df = pd.read_json(stored_data, orient="split")
    unique_values = df[filter_column].unique().tolist()

    return dbc.FormGroup(
        [
            dbc.Label(f"Select {filter_column} Value"),
            dcc.Dropdown(
                id="filter-value",
                options=[
                    {"label": str(val), "value": str(val)} for val in unique_values
                ],
                value=str(unique_values[0]) if unique_values else None,
            ),
        ]
    )


# Callback to update group value dropdown based on selected column
@app.callback(
    Output("group-value-container", "children"),
    [Input("group-column", "value")],
    [State("stored-data", "data")],
)
def update_group_value_dropdown(group_column, stored_data):
    if group_column == "None" or stored_data is None:
        return []

    df = pd.read_json(stored_data, orient="split")
    unique_values = df[group_column].unique().tolist()

    return dbc.FormGroup(
        [
            dbc.Label(f"Select {group_column} Value for Detailed Analysis"),
            dcc.Dropdown(
                id="group-value",
                options=[
                    {"label": str(val), "value": str(val)} for val in unique_values
                ],
                value=str(unique_values[0]) if unique_values else None,
            ),
        ]
    )


# Callback to show/hide Monte Carlo options
@app.callback(
    Output("monte-carlo-options", "children"), [Input("group-column", "value")]
)
def update_monte_carlo_options(group_column):
    if group_column == "None":
        return dbc.FormGroup(
            [
                dbc.Label("Number of Monte Carlo Samples"),
                dcc.Slider(
                    id="monte-carlo-samples",
                    min=100,
                    max=1000,
                    step=100,
                    value=500,
                    marks={i: str(i) for i in range(100, 1001, 100)},
                ),
            ]
        )
    return []


# Callback for Sensitivity Analysis plot and stats
@app.callback(
    [Output("sensitivity-plot", "figure"), Output("sensitivity-stats", "children")],
    [
        Input("x-variable", "value"),
        Input("y-variable", "value"),
        Input("filter-column", "value"),
        Input("filter-value", "value"),
    ],
    [State("stored-data", "data")],
)
def update_sensitivity_analysis(x_var, y_var, filter_col, filter_val, stored_data):
    if None in [x_var, y_var, stored_data]:
        return go.Figure(), []

    df = pd.read_json(stored_data, orient="split")

    # Apply filter if specified
    if filter_col != "None" and filter_val is not None:
        try:
            # Try to convert filter_val to original data type
            original_type = type(df[filter_col].iloc[0])
            converted_val = original_type(filter_val)
            filtered_df = df[df[filter_col] == converted_val]
        except:
            # If conversion fails, use as string
            filtered_df = df[df[filter_col].astype(str) == filter_val]
    else:
        filtered_df = df

    # Create scatter plot with trend line
    fig = px.scatter(
        filtered_df, x=x_var, y=y_var, trendline="ols", title=f"{y_var} vs {x_var}"
    )

    # Calculate statistics
    correlation = filtered_df[[x_var, y_var]].corr().iloc[0, 1]

    stats_table = dbc.Table.from_dataframe(
        filtered_df[[x_var, y_var]]
        .describe()
        .reset_index()
        .rename(columns={"index": "Statistic"}),
        striped=True,
        bordered=True,
        hover=True,
    )

    stats_content = html.Div(
        [
            html.H5("Statistical Analysis"),
            html.P(f"Correlation coefficient: {correlation:.4f}"),
            html.H6("Summary Statistics:"),
            stats_table,
        ]
    )

    return fig, stats_content


# Callback for Uncertainty Analysis plot and stats
@app.callback(
    [Output("uncertainty-plot", "figure"), Output("uncertainty-stats", "children")],
    [
        Input("parameter-select", "value"),
        Input("group-column", "value"),
        Input("group-value", "value"),
        Input("monte-carlo-samples", "value"),
    ],
    [State("stored-data", "data")],
)
def update_uncertainty_analysis(
    parameter, group_col, group_val, n_samples, stored_data
):
    if None in [parameter, stored_data]:
        return go.Figure(), []

    df = pd.read_json(stored_data, orient="split")

    if group_col != "None":
        # Group analysis
        if group_val is not None:
            try:
                # Try to convert group_val to original data type
                original_type = type(df[group_col].iloc[0])
                converted_val = original_type(group_val)
                filtered_df = df[df[group_col] == converted_val]
            except:
                # If conversion fails, use as string
                filtered_df = df[df[group_col].astype(str) == group_val]

            # Create histogram for the selected group
            fig = px.histogram(
                filtered_df,
                x=parameter,
                marginal="box",
                title=f"Distribution of {parameter} for {group_col}={group_val}",
            )

            # Statistics for selected group
            stats_df = (
                filtered_df[parameter]
                .describe()
                .reset_index()
                .rename(columns={"index": "Statistic", parameter: "Value"})
            )

            stats_content = html.Div(
                [
                    html.H5(f"Statistics for {group_col}={group_val}"),
                    dbc.Table.from_dataframe(
                        stats_df, striped=True, bordered=True, hover=True
                    ),
                ]
            )

        else:
            # Create boxplot by group
            fig = px.box(
                df,
                x=group_col,
                y=parameter,
                title=f"Distribution of {parameter} by {group_col}",
            )

            # Statistics by group
            group_stats = (
                df.groupby(group_col)[parameter]
                .agg(["mean", "std", "min", "max"])
                .reset_index()
            )

            stats_content = html.Div(
                [
                    html.H5("Statistics by Group"),
                    dbc.Table.from_dataframe(
                        group_stats, striped=True, bordered=True, hover=True
                    ),
                ]
            )
    else:
        # Overall distribution analysis
        fig = px.histogram(
            df, x=parameter, marginal="box", title=f"Distribution of {parameter}"
        )

        # Calculate basic statistics
        stats_df = (
            df[parameter]
            .describe()
            .reset_index()
            .rename(columns={"index": "Statistic", parameter: "Value"})
        )

        # For Monte Carlo simulation
        if n_samples:
            mean = df[parameter].mean()
            std = df[parameter].std()

            # Generate simulated data
            simulated = np.random.normal(mean, std, n_samples)

            # Add second plot for Monte Carlo
            fig2 = px.histogram(
                x=simulated,
                marginal="box",
                title=f"Monte Carlo Simulation for {parameter} (n={n_samples})",
            )

            # Add vertical lines for mean and confidence intervals
            fig2.add_vline(
                x=mean,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean:.4f}",
                annotation_position="top right",
            )
            fig2.add_vline(
                x=mean + 1.96 * std,
                line_dash="dash",
                line_color="green",
                annotation_text=f"95% CI Upper: {mean + 1.96*std:.4f}",
                annotation_position="top right",
            )
            fig2.add_vline(
                x=mean - 1.96 * std,
                line_dash="dash",
                line_color="green",
                annotation_text=f"95% CI Lower: {mean - 1.96*std:.4f}",
                annotation_position="top left",
            )

            # Stats content with both tables
            stats_content = html.Div(
                [
                    html.Div(
                        [
                            html.H5("Summary Statistics"),
                            dbc.Table.from_dataframe(
                                stats_df, striped=True, bordered=True, hover=True
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.H5("Monte Carlo Simulation Results"),
                            html.P(
                                f"95% Confidence Interval: [{mean - 1.96*std:.4f}, {mean + 1.96*std:.4f}]"
                            ),
                            dcc.Graph(figure=fig2),
                        ]
                    ),
                ]
            )
        else:
            # Stats content with just the main table
            stats_content = html.Div(
                [
                    html.H5("Summary Statistics"),
                    dbc.Table.from_dataframe(
                        stats_df, striped=True, bordered=True, hover=True
                    ),
                ]
            )

    return fig, stats_content


# Run the server
if __name__ == "__main__":
    app.run(debug=True)
