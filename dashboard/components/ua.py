"""Component to display uncertainity data."""

from typing import Any
from dash import dcc, html, dash_table
import pandas as pd

from .data import (
    SECTOR_DROPDOWN_OPTIONS,
    RESULT_TYPE_DROPDOWN_OPTIONS,
    SECTOR_MAPPER_CACHE,
    RESULT_MAPPER_CACHE,
    UA_RESULT_NICE_NAMES,
)

from .styles import DATA_TABLE_STYLE
from .utils import (
    DEFAULT_PLOTLY_THEME,
    DEFAULT_HEIGHT,
    DEFAULT_LEGEND,
    DEFAULT_OPACITY,
    get_emission_limits,
    get_y_label,
    DEFAULT_2005_EMISSION_LIMIT,
    DEFAULT_2030_EMISSION_LIMIT,
)
from . import ids as ids
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde


import logging

logger = logging.getLogger(__name__)


def get_stores_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts out stores that will have different units.

    This is ALSO SUUUUUUUPER hacky, but this just NEEDS to get done! :|
    """
    return df[
        (df.result.str.contains("Water Heater")) | (df.result.str.contains("Battery"))
    ]


def ua_options_block() -> html.Div:
    """UA options block component."""
    return html.Div(
        [
            ua_result_type_dropdown(),
            ua_result_sector_dropdown(),
            ua_percentile_interval_slider(),
            ua_emission_target_rb(),
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
                value="cost",
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


def ua_percentile_interval_slider() -> html.Div:
    """UA slider component."""

    return html.Div(
        [
            html.H6("Percentile Interval:"),
            dcc.RangeSlider(
                id=ids.UA_INTERVAL_SLIDER,
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
                updatemode="mouseup",
            ),
        ],
    )


def ua_emission_target_rb() -> html.Div:
    """UA emission target rb component."""
    return html.Div(
        [
            html.H6("Show Emission Target"),
            dcc.RadioItems(
                id=ids.UA_EMISSION_TARGET_RB,
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


def _filter_ua_on_result_sector(
    df: pd.DataFrame, sector: str, metadata: dict
) -> pd.DataFrame:
    """Filter UA data on result sector."""

    if sector == "all":
        return df

    # Use pre-computed cache: {sector_label -> [result_value, ...]}
    result_values = SECTOR_MAPPER_CACHE.get(sector, [])
    keep = ["run", "state"] + [v for v in result_values if v in df.columns]
    return df[[c for c in keep if c in df.columns]]


def _filter_ua_on_result_type(
    df: pd.DataFrame, result_type: str, metadata: dict
) -> pd.DataFrame:
    """Filter UA data on result type."""

    if df.empty:
        logger.debug("No UA data found")
        return df

    # Use pre-computed cache: {result_type_label -> [result_value, ...]}
    cols = list(set(RESULT_MAPPER_CACHE.get(result_type, [])))

    # marginal cost is an edge case when in the UA page we only want to show the average
    if result_type == "marginal_cost":
        cols = [x for x in cols if x.endswith(("_elec", "_energy", "_gas", "_ind", "_srvc", "_trn", "_res", "_com"))]

    if not cols:
        logger.debug(f"No columns found for result type {result_type}")
        return pd.DataFrame(index=df.index)
    else:
        return df.set_index("run")[[x for x in cols if x in df]].reset_index()


def remove_ua_outliers(df: pd.DataFrame, interval: list[int]) -> pd.DataFrame:
    """Replace UA outliers with NaN values (values outside the interval are set to NaN)."""
    # intervals defined as ints, but pandas quantile expects [0,1]
    interval_low = round(min(interval) / 100, 2)
    interval_high = round(max(interval) / 100, 2)
    logger.debug(
        f"Removing UA outliers with interval {interval_low} and {interval_high}"
    )

    # Only clip numeric columns that are not the run index or state label
    non_numeric = [c for c in df.columns if c in ("run", "state")]
    numeric_cols = df.select_dtypes(include="number").columns.difference(non_numeric)

    df = df.copy()
    if len(numeric_cols):
        lo = df[numeric_cols].quantile(interval_low)
        hi = df[numeric_cols].quantile(interval_high)
        df[numeric_cols] = df[numeric_cols].where(
            df[numeric_cols].ge(lo) & df[numeric_cols].le(hi)
        )

    return df


def _read_serialized_ua_data(data: dict[str, Any]) -> pd.DataFrame:
    """Read serialized UA data from the dash store."""
    df = pd.DataFrame(data)
    # logger.debug(f"Searlized UA data read: {df}")
    df["run"] = df.run.astype(int)
    return df.set_index("run")


def _apply_nice_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply nice names."""
    logger.debug("Applying nice names")
    return df.rename(columns=UA_RESULT_NICE_NAMES)


def _melt_results(df: pd.DataFrame) -> pd.DataFrame:
    """Melt results from UA data table."""
    return df.melt(id_vars=["run"], var_name="result", value_name="value")


def filter_ua_on_result_sector_and_type(
    df: pd.DataFrame, result_sector: str, result_type: str, metadata: dict
) -> pd.DataFrame:
    """Filter UA data on result sector and type."""
    filtered_on_sector = _filter_ua_on_result_sector(df, result_sector, metadata)
    filtered_on_sector_and_type = _filter_ua_on_result_type(
        filtered_on_sector, result_type, metadata
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

    if nice_names:
        df = _apply_nice_names(df)

    # Format columns for display
    columns = [
        {"name": "run", "id": "run", "type": "numeric", "format": {"specifier": ".0f"}}
    ] + [
        {"name": col, "id": col, "type": "numeric", "format": {"specifier": ".3f"}}
        for col in df.columns
        if col != "param"
    ]

    df = df.reset_index().fillna(0)

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=columns,
        **DATA_TABLE_STYLE,
    )


def get_ua_scatter_plot(
    data: dict[str, Any],
    nice_names: bool = True,
    emissions: list[dict[str, Any]] | None = None,
    **kwargs,
) -> go.Figure:
    """UA scatter plot component."""

    if not data:
        logger.debug("No UA scatter plot data found")
        return px.scatter(pd.DataFrame(columns=["run", "value"]), x="run", y="value")

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    # drop outlier data
    df_melted = df_melted.dropna(subset=["value"])

    # Prune data to avoid huge dash payloads
    if len(df_melted) > 1000:
        logger.debug(
            f"Downsampling UA scatter plot from {len(df_melted)} to 1000 points"
        )
        df_melted = df_melted.sample(n=1000, random_state=42)

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    result_type = kwargs.get("result_type", None)
    ylabel = get_y_label(df_melted, result_type)

    df_stores = get_stores_df(df_melted)
    if not df_stores.empty:
        df_melted = df_melted[~df_melted.result.isin(df_stores.result)].copy()

    fig = px.scatter(
        df_melted,
        x="run",
        y="value",
        color="result",
        labels=dict(run="Run Number", value=ylabel, result="Result Type"),
    )

    fig.update_layout(
        title="",
        xaxis_title="Run Number",
        yaxis_title=ylabel,
        height=DEFAULT_HEIGHT,
        showlegend=True,
        legend=DEFAULT_LEGEND,
        template=color_theme,
    )

    # SUUUUPER hacky way to account for stores being in different units
    if not df_stores.empty:
        y_label_store = ylabel.replace("(MW)", "(MWh)")

        scatter_fig = px.scatter(
            df_stores,
            x="run",
            y="value",
            color="result",
            labels=dict(run="Run Number", value=y_label_store, result="Result Type"),
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        # so the stores are clearly different
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


def get_ua_barchart(
    data: dict[str, Any],
    nice_names: bool = True,
    emissions: list[dict[str, Any]] | None = None,
    **kwargs,
) -> go.Figure:
    """UA barchart component showing mean values with 95% confidence intervals for each result category."""
    if not data:
        logger.debug("No UA barchart data found")
        return px.bar(pd.DataFrame(columns=["result", "value"]), x="result", y="value")

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    df_melted = df_melted.dropna(subset=["value"])

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    result_type = kwargs.get("result_type", None)
    ylabel = get_y_label(df_melted, result_type)

    df_stores = get_stores_df(df_melted)
    if not df_stores.empty:
        df_melted = df_melted[~df_melted.result.isin(df_stores.result)].copy()

    # Calculate mean and std for each result category
    stats_df = (
        df_melted.groupby("result")
        .agg(value=("value", "mean"), std=("value", "std"))
        .reset_index()
    )

    fig = px.bar(
        stats_df,
        x="result",
        y="value",
        error_y="std",
        labels=dict(result="", value=ylabel),
        opacity=DEFAULT_OPACITY,
    )

    # Update layout
    fig.update_layout(
        title="(Mean ± 95% CI)",
        showlegend=False,
        height=DEFAULT_HEIGHT,
        bargap=0.2,
        xaxis=dict(tickangle=45),
        template=color_theme,
    )

    if not df_stores.empty:
        y_label_store = ylabel.replace("(MW)", "(MWh)")

        stats_df_stores = (
            df_stores.groupby("result")
            .agg(value=("value", "mean"), std=("value", "std"))
            .reset_index()
        )

        bar_fig = px.bar(
            stats_df_stores,
            x="result",
            y="value",
            error_y="std",
            labels=dict(result="", value=y_label_store),
            opacity=DEFAULT_OPACITY,
        )

        # vertical line to separate MWh from MW data
        fig.add_vline(
            x=len(stats_df) - 0.5,  # Position before the stores data
            line_dash="solid",
            line_color="black",
            line_width=2,
            opacity=0.8,
        )

        for trace in bar_fig.data:
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


def get_ua_histogram(
    data: dict[str, Any], nice_names: bool = True, **kwargs
) -> html.Div:
    """UA histogram component with overlaid probability density functions"""
    if not data:
        logger.debug("No UA histogram data found")
        return html.Div(
            [dcc.Graph(figure=px.histogram(pd.DataFrame(columns=["value"]), x="value"))]
        )

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    ylabel = kwargs.get("result_type", None)

    result_types = df_melted["result"].unique()
    logger.debug(f"Historgram result types: {result_types}")
    fig_height = (
        int(DEFAULT_HEIGHT * (2 / 3)) if len(result_types) > 1 else DEFAULT_HEIGHT
    )

    figures = []

    for result_type in result_types:
        result_data = df_melted[df_melted["result"] == result_type]

        # Calculate number of bins using Freedman-Diaconis rule

        fig = px.histogram(
            result_data,
            x="value",
            labels=dict(value=ylabel),
            opacity=DEFAULT_OPACITY,
            title=f"{result_type}",
        )

        fig.update_layout(
            yaxis=dict(title="Count"),
            yaxis2=dict(
                # title="Probability Density",
                overlaying="y",
                side="right",
                showgrid=False,
                showticklabels=False,
            ),
        )

        # Add PDF overlay
        values = result_data["value"].dropna()

        # Check if we have enough data and variance for KDE
        # this will not be the case if all values are the same for all model runs
        if len(values) > 1 and values.var() > 1e-10:
            try:
                kde = gaussian_kde(values)
                x_range = np.linspace(
                    result_data["value"].min(), result_data["value"].max(), 100
                )
                y_range = kde(x_range)

                # Verify that the KDE integrates to approximately 1
                dx = x_range[1] - x_range[0]
                area = np.sum(y_range * dx)
                logger.debug(f"KDE area under curve with dx={dx} is: {area:.6f}")

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        name="PDF",
                        line=dict(width=2),
                        showlegend=True,
                        yaxis="y2",
                    )
                )
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.warning(f"Could not compute KDE for {result_type}: {e}")
        else:
            logger.debug(f"Insufficient variance in data for KDE in {result_type}")

        fig.update_layout(
            height=fig_height,
            showlegend=True,
            legend=DEFAULT_LEGEND,
            template=color_theme,
            margin=dict(t=50, l=50, r=50, b=50),
        )

        figures.append(dcc.Graph(figure=fig))

    if len(result_types) > 1:
        return html.Div(
            figures,
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, 1fr)",
                "gap": "20px",
                "padding": "20px",
            },
            id=ids.UA_HISTOGRAM,
        )
    else:
        return html.Div(
            figures[0],
            id=ids.UA_HISTOGRAM,
        )


def get_ua_violin_plot(
    data: dict[str, Any],
    nice_names: bool = True,
    emissions: list[dict[str, Any]] | None = None,
    **kwargs,
) -> go.Figure:
    """UA violin plot component."""
    if not data:
        logger.debug("No UA violin plot data found")
        return px.violin(
            pd.DataFrame(columns=["result", "value"]), x="result", y="value"
        )

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    # Prune data to avoid huge dash payloads
    if len(df_melted) > 1000:
        logger.debug(f"Downsampling from {len(df_melted)} to 1000 points")
        df_melted = df_melted.sample(n=1000, random_state=42)

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    result_type = kwargs.get("result_type", None)
    ylabel = get_y_label(df_melted, result_type)

    df_stores = get_stores_df(df_melted)
    if not df_stores.empty:
        df_melted = df_melted[~df_melted.result.isin(df_stores.result)].copy()

    fig = px.violin(
        df_melted,
        x="result",
        y="value",
        labels=dict(result="", value=ylabel),
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

    if not df_stores.empty:
        y_label_store = ylabel.replace("(MW)", "(MWh)")

        violin_fig = px.violin(
            df_stores,
            x="result",
            y="value",
            labels=dict(result="", value=y_label_store),
        )

        # vertical line to separate MWh from MW data
        fig.add_vline(
            x=len(df_melted["result"].unique())
            - 0.5,  # Position before the stores data
            line_dash="solid",
            line_color="black",
            line_width=2,
            opacity=0.8,
        )

        for trace in violin_fig.data:
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


def get_ua_box_whisker(
    data: dict[str, Any],
    nice_names: bool = True,
    emissions: list[dict[str, Any]] | None = None,
    **kwargs,
) -> go.Figure:
    """UA box whisker plot component."""
    if not data:
        logger.debug("No UA box whisker plot data found")
        return px.box(pd.DataFrame(columns=["result", "value"]), x="result", y="value")

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    # Prune data to avoid huge dash payloads
    if len(df_melted) > 1000:
        logger.debug(f"Downsampling from {len(df_melted)} to 1000 points")
        df_melted = df_melted.sample(n=1000, random_state=42)

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    result_type = kwargs.get("result_type", None)
    ylabel = get_y_label(df_melted, result_type)

    df_stores = get_stores_df(df_melted)
    if not df_stores.empty:
        df_melted = df_melted[~df_melted.result.isin(df_stores.result)].copy()

    fig = px.box(
        df_melted,
        x="result",
        y="value",
        labels=dict(result="", value=ylabel),
    )

    # Update layout
    fig.update_layout(
        title="",
        showlegend=False,
        height=DEFAULT_HEIGHT,
        xaxis=dict(tickangle=45),
        template=color_theme,
    )

    if not df_stores.empty:
        y_label_store = ylabel.replace("(MW)", "(MWh)")

        box_fig = px.box(
            df_stores,
            x="result",
            y="value",
            labels=dict(result="", value=y_label_store),
        )

        # vertical line to separate MWh from MW data
        fig.add_vline(
            x=len(df_melted["result"].unique())
            - 0.5,  # Position before the stores data
            line_dash="solid",
            line_color="black",
            line_width=2,
            opacity=0.8,
        )

        for trace in box_fig.data:
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
