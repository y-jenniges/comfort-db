"""Dash application for exploring COMFORT parameter data.

Three modes:
    - **Scatter**: browse raw observations with spatial, depth and temporal filters.
    - **Grid**: inspect an existing wide grid table (produced by GridManager).
    - **CSV**: inspect an existing gridded table form a CSV.

Set environment variables before launching:
    COMFORT_DB_PATH - path to the COMFORT SQLite database

Usage:
    python -m comfort.app.explorer

or from Python:
    from comfort.app.explorer import create_app
    create_app("/path/to/comfort.sqlite").run(debug=True)
"""
from __future__ import annotations

import logging
import os
import sqlite3
import base64
import io
import numpy as np
import pandas as pd

from ..util.optional import import_optional

# Fail with install instructions if the dash extra is missing
for _module in ("dash", "dash_bootstrap_components", "plotly"):
    import_optional(_module, "dash", "the COMFORT app")

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ..database.information import get_names_of_all_parameter_tables
from ..io import list_parameters, read_parameter
from ..qc import QC_GOOD
from ..units import UnitsConverter

logger = logging.getLogger(__name__)

# Database path, set by create_app()
_db_path: str | None = None

# Callbacks collected by _register_callback (so that multiple apps can run simultaneously)
_CALLBACKS: list = []


def _register_callback(*args, **kwargs):
    """Collect a callback, which create_app() can attach to."""
    def decorator(func):
        _CALLBACKS.append((args, kwargs, func))
        return func
    return decorator

# Server-side caches (single-user local app)
_scatter_cache: dict = {"df": None, "bounds": None, "unit_note": ""}
_grid_cache: dict = {"depths": [], "times": []}
_csv_cache: dict = {"df": None, "depths": [], "times": []}


def _clear_scatter_cache() -> None:
    """Release the cached scatter DataFrame to free memory."""
    _scatter_cache["df"] = None
    _scatter_cache["bounds"] = None
    _scatter_cache["unit_note"] = ""


def _clear_csv_cache() -> None:
    """Release the cached CSV DataFrame to free memory."""
    _csv_cache["df"] = None
    _csv_cache["depths"] = []
    _csv_cache["times"] = []


# Shared map styling
_GEO_STYLE = dict(
    projection_type="natural earth",
    showland=True, landcolor="#e0e0e0",
    showocean=True, oceancolor="#f7fbff",
    showcoastlines=True, coastlinecolor="#888888",
    showlakes=True, lakecolor="#f7fbff",
)


def _get_conn() -> sqlite3.Connection:
    """Open a connection to the COMFORT database."""
    if _db_path is None:
        raise RuntimeError("Database path is not set. Create the app with create_app().")
    return sqlite3.connect(_db_path)


def _query_parameters() -> list[str]:
    """Return available parameter names from the database."""
    conn = _get_conn()
    try:
        return list_parameters(conn)
    finally:
        conn.close()


def _query_wide_tables() -> list[str]:
    """Return names of existing wide grid tables."""
    conn = _get_conn()
    try:
        return get_names_of_all_parameter_tables(
            conn, like_pattern="wide|_%", escape_char="|",
            include_digits=True, table_type="table",
        )
    finally:
        conn.close()


def _query_cruise_labels(conn: sqlite3.Connection) -> dict:
    """Map CRUISE.ID to a human-readable label (EXPOCODE, falling back to
    CRUISE_NUMBER). Returns {} if no CRUISE table exists in this database."""
    try:
        df = pd.read_sql_query("SELECT ID, EXPOCODE, CRUISE_NUMBER FROM CRUISE", conn)
    except Exception:
        return {}
    has_expocode = df["EXPOCODE"].notna() & (df["EXPOCODE"].astype(str) != "")
    label = df["EXPOCODE"].where(has_expocode, df["CRUISE_NUMBER"].astype(str))
    return dict(zip(df["ID"], label))


def _format_hover_list(series, max_items=5):
    """Format up to max_items distinct values from a Series, '...' if more."""
    uniq = sorted({str(v) for v in series.dropna()})
    if not uniq:
        return "-"
    if len(uniq) > max_items:
        return ", ".join(uniq[:max_items]) + f", ... (+{len(uniq) - max_items} more)"
    return ", ".join(uniq)


def _make_slider_marks(positions, first_label, last_label):
    """Ticks at every position; labels only at first and last step."""
    marks = {p: "" for p in positions}
    if positions:
        marks[positions[0]] = first_label
        marks[positions[-1]] = last_label
    return marks


def _format_range_label(lo, hi, fmt="{}"):
    """Format a range as a single value, or 'lo-hi' when the bounds differ."""
    lo_s, hi_s = fmt.format(lo), fmt.format(hi)
    return lo_s if lo_s == hi_s else f"{lo_s}-{hi_s}"


# (min, max, value, marks, step, disabled) for an empty/disabled range slider
_EMPTY_RANGE_SLIDER = (0, 1, [0, 1], {0: "-"}, None, True)

# Scatter "Colour by" options
_COLOUR_BY_LABELS = {
    "VAL": None,
    "LEV_M": "Depth (m)",
    "PQF1": "QC Flag 1",
    "PQF2": "QC Flag 2",
    "SQF": "Station QC Flag",
    "INSTRUMENT_ID": "Instrument ID",
    "_year": "Year",
}


def _build_layout(all_params: list[str]) -> dbc.Container:
    """Construct the Dash layout with scatter and grid tabs."""
    return dbc.Container([
        dbc.Row(dbc.Col(
            html.H2("COMFORT Data Explorer", className="text-center my-3"),
        )),

        dbc.Row([

            # Left panel - controls
            dbc.Col([
                dbc.Tabs(id="tabs", active_tab="scatter", children=[
                    _scatter_tab(all_params),
                    _grid_tab(),
                    _csv_tab(),
                ]),
            ], md=5, lg=4),

            # Right panel - chart and status
            dbc.Col([
                dbc.Alert(
                    id="alert", is_open=False,
                    dismissable=True, color="danger",
                ),
                dcc.Loading(type="default", children=[
                    dcc.Graph(id="chart", style={"height": "75vh"}),
                    html.Div(id="status",
                             className="text-muted mt-2 text-center"),
                ]),
            ], md=7, lg=8),
        ]),
    ], fluid=True)


def _scatter_tab(all_params: list[str]) -> dbc.Tab:
    """Build the Scatter tab layout."""
    return dbc.Tab(label="Scatter", tab_id="scatter", children=[
        html.Div(className="pt-3", children=[

            # Parameter selector
            dbc.Card([
                dbc.CardHeader("Parameter"),
                dbc.CardBody(dbc.Select(
                    id="scatter-param",
                    options=[{"label": p, "value": p} for p in all_params],
                    value=all_params[0] if all_params else None,
                )),
            ], className="mb-3"),

            # Spatial extent
            dbc.Card([
                dbc.CardHeader("Spatial extent"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([dbc.Label("Lon min"), dbc.Input(
                            id="sc-lon-min", type="number",
                            min=-180, max=180, value=-77,
                        )], width=6),
                        dbc.Col([dbc.Label("Lon max"), dbc.Input(
                            id="sc-lon-max", type="number",
                            min=-180, max=180, value=30,
                        )], width=6),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([dbc.Label("Lat min"), dbc.Input(
                            id="sc-lat-min", type="number",
                            min=-90, max=90, value=0,
                        )], width=6),
                        dbc.Col([dbc.Label("Lat max"), dbc.Input(
                            id="sc-lat-max", type="number",
                            min=-90, max=90, value=70,
                        )], width=6),
                    ]),
                ]),
            ], className="mb-3"),

            # QC filter
            dbc.Card([
                dbc.CardHeader("Options"),
                dbc.CardBody(
                    dbc.Switch(id="sc-qc", value=True, label="QC_GOOD"),
                ),
            ], className="mb-3"),

            dbc.Button(
                "Load data", id="btn-scatter",
                color="primary", size="lg", className="w-100 mb-3",
            ),

            # Depth range slider
            dbc.Card([
                dbc.CardHeader("Depth range (m)"),
                dbc.CardBody(dcc.RangeSlider(
                    id="sc-depth-slider",
                    min=0, max=6000, value=[0, 6000], step=10,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                )),
            ], className="mb-3"),

            # Year range slider
            dbc.Card([
                dbc.CardHeader("Year range"),
                dbc.CardBody(dcc.RangeSlider(
                    id="sc-year-slider",
                    min=1900, max=2025, value=[1900, 2025], step=1,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                )),
            ], className="mb-3"),

            # Colour by (map only)
            dbc.Card([
                dbc.CardHeader("Colour by (map)"),
                dbc.CardBody(dbc.Select(
                    id="sc-colour-by",
                    options=[
                        {"label": "Value (selected parameter)", "value": "VAL"},
                        {"label": "Depth (m)", "value": "LEV_M"},
                        {"label": "QC Flag 1", "value": "PQF1"},
                        {"label": "QC Flag 2", "value": "PQF2"},
                        {"label": "Station QC Flag", "value": "SQF"},
                        {"label": "Instrument ID", "value": "INSTRUMENT_ID"},
                        {"label": "Year", "value": "_year"},
                    ],
                    value="VAL",
                )),
            ], className="mb-3"),

            # Plot type
            dbc.Card([
                dbc.CardHeader("Plot type"),
                dbc.CardBody(dbc.RadioItems(
                    id="scatter-plot-type",
                    options=[
                        {"label": "Map (lat/lon)", "value": "map"},
                        {"label": "Profile (depth)", "value": "profile"},
                        {"label": "Histogram", "value": "hist"},
                        {"label": "Time series", "value": "time"},
                    ],
                    value="map", inline=True,
                )),
            ], className="mb-3"),

            # Global vs cropped checkbox
            dbc.Checkbox(
                id="sc-global-map",
                value=False,
                label="Show global",
            )
        ]),
    ])


def _grid_tab() -> dbc.Tab:
    """Build the Grid tab layout."""
    return dbc.Tab(label="Grid", tab_id="grid", children=[
        html.Div(className="pt-3", children=[

            # Wide table selector
            dbc.Card([
                dbc.CardHeader("Wide table"),
                dbc.CardBody([
                    dbc.Select(id="grid-table", options=[]),
                    dbc.Button(
                        "Refresh", id="btn-refresh-grid",
                        size="sm", color="secondary", className="mt-2",
                    ),
                ]),
            ], className="mb-3"),

            # Parameter column
            dbc.Card([
                dbc.CardHeader("Parameter column"),
                dbc.CardBody(dbc.Select(id="grid-param", options=[])),
            ], className="mb-3"),

            # Depth range slider, a widened range averages on the map
            dbc.Card([
                dbc.CardHeader("Depth range (m)"),
                dbc.CardBody([
                    dcc.RangeSlider(
                        id="grid-depth-slider",
                        min=0, max=1, value=[0, 1], step=None,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(
                        id="grid-depth-display",
                        className="text-center text-muted mt-1"
                    )
                ])
            ], className="mb-3"),

            # Time range slider (discrete steps, index-based)
            dbc.Card([
                dbc.CardHeader("Time range"),
                dbc.CardBody([
                    dcc.RangeSlider(
                        id="grid-time-slider",
                        min=0, max=1, value=[0, 1], step=None,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(
                        id="grid-time-display",
                        className="text-center text-muted mt-1"
                    )
                ])
            ], className="mb-3"),

            # Plot type
            dbc.Card([
                dbc.CardHeader("Plot type"),
                dbc.CardBody(dbc.RadioItems(
                    id="grid-plot-type",
                    options=[
                        {"label": "Map (lat/lon)", "value": "map"},
                        {"label": "Histogram", "value": "hist"},
                        {"label": "Missing values", "value": "missing"},
                    ],
                    value="map", inline=True,
                )),
            ], className="mb-3"),

            # Global vs cropped checkbox
            dbc.Checkbox(
                id="grid-global-map",
                value=False,
                label="Show global",
            )
        ]),
    ])


def _csv_tab() -> dbc.Tab:
    """Build the CSV tab layout."""
    return dbc.Tab(label="CSV", tab_id="csv", children=[
        html.Div(className="pt-3", children=[

            # CSV selector
            dbc.Card([
                dbc.CardHeader("CSV file"),
                dbc.CardBody([
                    dcc.Upload(
                        id="csv-upload",
                        children=dbc.Button(
                            "Browse...",
                            color="secondary",
                            className="mb-2",
                        ),
                        multiple=False,
                    ),
                    html.Div(
                        id="csv-filename",
                        className="text-muted small mt-2",
                        children="No file selected.",
                    ),
                ]),
            ], className="mb-3"),

            # Parameter column
            dbc.Card([
                dbc.CardHeader("Parameter column"),
                dbc.CardBody(dbc.Select(id="csv-param", options=[])),
            ], className="mb-3"),

            # Depth range slider (discrete levels)
            dbc.Card([
                dbc.CardHeader("Depth range (m)"),
                dbc.CardBody([
                    dcc.RangeSlider(
                        id="csv-depth-slider",
                        min=0, max=1, value=[0, 1], step=None,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(
                        id="csv-depth-display",
                        className="text-center text-muted mt-1"
                    )
                ])
            ], className="mb-3"),

            # Time range slider (discrete steps, index-based)
            dbc.Card([
                dbc.CardHeader("Time range"),
                dbc.CardBody([
                    dcc.RangeSlider(
                        id="csv-time-slider",
                        min=0, max=1, value=[0, 1], step=None,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(
                        id="csv-time-display",
                        className="text-center text-muted mt-1"
                    )
                ])
            ], className="mb-3"),

            # Plot type
            dbc.Card([
                dbc.CardHeader("Plot type"),
                dbc.CardBody(dbc.RadioItems(
                    id="csv-plot-type",
                    options=[
                        {"label": "Map (lat/lon)", "value": "map"},
                        {"label": "Histogram", "value": "hist"},
                        {"label": "Missing values", "value": "missing"},
                    ],
                    value="map", inline=True,
                )),
            ], className="mb-3"),

            # Global vs cropped checkbox
            dbc.Checkbox(
                id="csv-global-map",
                value=False,
                label="Show global",
            )
        ]),
    ])


# --- Tab switch -------------------------------------------------------------- #


# Reset every mode's cache, chart and controls on every tab switch
@_register_callback(
    Output("chart", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("alert", "is_open", allow_duplicate=True),
    # Scatter controls
    Output("sc-depth-slider", "min", allow_duplicate=True),
    Output("sc-depth-slider", "max", allow_duplicate=True),
    Output("sc-depth-slider", "value", allow_duplicate=True),
    Output("sc-year-slider", "min", allow_duplicate=True),
    Output("sc-year-slider", "max", allow_duplicate=True),
    Output("sc-year-slider", "value", allow_duplicate=True),
    Output("sc-colour-by", "value", allow_duplicate=True),
    # CSV controls
    Output("csv-upload", "contents", allow_duplicate=True),
    Output("csv-upload", "filename", allow_duplicate=True),
    Output("csv-filename", "children", allow_duplicate=True),
    Output("csv-param", "options", allow_duplicate=True),
    Output("csv-param", "value", allow_duplicate=True),
    Output("csv-depth-slider", "min", allow_duplicate=True),
    Output("csv-depth-slider", "max", allow_duplicate=True),
    Output("csv-depth-slider", "value", allow_duplicate=True),
    Output("csv-depth-slider", "marks", allow_duplicate=True),
    Output("csv-depth-slider", "step", allow_duplicate=True),
    Output("csv-depth-slider", "disabled", allow_duplicate=True),
    Output("csv-time-slider", "min", allow_duplicate=True),
    Output("csv-time-slider", "max", allow_duplicate=True),
    Output("csv-time-slider", "value", allow_duplicate=True),
    Output("csv-time-slider", "marks", allow_duplicate=True),
    Output("csv-time-slider", "step", allow_duplicate=True),
    Output("csv-time-slider", "disabled", allow_duplicate=True),
    Input("tabs", "active_tab"),
    prevent_initial_call=True,
)
def _on_tab_switch(active_tab):
    """Reset every mode's cache, chart and controls when switching tabs."""
    _clear_scatter_cache()
    _clear_csv_cache()

    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white")

    return (
        empty_fig, "", False,
        0, 6000, [0, 6000],
        1900, 2025, [1900, 2025],
        "VAL",
        None, None, "No file selected.", [], None,
        *_EMPTY_RANGE_SLIDER,
        *_EMPTY_RANGE_SLIDER,
    )


# --- Scatter callbacks ---------------------------------------------------------------


# Single scatter callback: load on button click, re-render on slider/plot change
@_register_callback(
    Output("chart", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("sc-depth-slider", "min"),
    Output("sc-depth-slider", "max"),
    Output("sc-depth-slider", "value"),
    Output("sc-year-slider", "min"),
    Output("sc-year-slider", "max"),
    Output("sc-year-slider", "value"),
    Output("alert", "children", allow_duplicate=True),
    Output("alert", "is_open", allow_duplicate=True),
    Input("btn-scatter", "n_clicks"),
    Input("sc-depth-slider", "value"),
    Input("sc-year-slider", "value"),
    Input("scatter-plot-type", "value"),
    Input("sc-global-map", "value"),
    Input("sc-colour-by", "value"),
    State("scatter-param", "value"),
    State("sc-lon-min", "value"), State("sc-lon-max", "value"),
    State("sc-lat-min", "value"), State("sc-lat-max", "value"),
    State("sc-qc", "value"),
    State("sc-depth-slider", "min"), State("sc-depth-slider", "max"),
    State("sc-year-slider", "min"), State("sc-year-slider", "max"),
    prevent_initial_call=True,
)
def _scatter_callback(n_clicks, depth_range, year_range, plot_type, show_global, colour_by,
                      param, lon_min, lon_max, lat_min, lat_max, use_qc,
                      cur_z_min, cur_z_max, cur_y_min, cur_y_max):
    """Load data on button click, or re-render from cache on slider change."""
    trigger = ctx.triggered_id
    no_slider = (no_update,) * 6

    # Button click: load data from database
    if trigger == "btn-scatter":
        if not param:
            return no_update, no_update, *no_slider, "Select a parameter.", True

        # Free previous data before loading new
        _clear_scatter_cache()

        try:
            conn = _get_conn()
            qc = QC_GOOD if use_qc else None

            # Load all rows matching the spatial and QC filters
            df = read_parameter(
                conn, param, quality_flags=qc,
                lon_min=lon_min, lon_max=lon_max,
                lat_min=lat_min, lat_max=lat_max,
                extra_station_cols=["CRUISE_ID", "ST_NUMBER_ORIGIN"],
            )

            # Convert units to the default for this parameter
            converted = False
            if "UNITS_ID" in df.columns and df["UNITS_ID"].nunique() > 1:
                try:
                    uc = UnitsConverter.from_connection(conn)
                    df = uc.convert_dataframe(df, param)
                    converted = True
                except (ValueError, Exception) as e:
                    logger.warning("Unit conversion skipped: %s", e)

            # Station/cruise hover labels
            if "ST_NUMBER_ORIGIN" in df.columns:
                df["_station_label"] = df["ST_NUMBER_ORIGIN"].where(
                    df["ST_NUMBER_ORIGIN"].notna(), df["ID"].astype(str),
                )
            if "CRUISE_ID" in df.columns:
                cruise_labels = _query_cruise_labels(conn)
                df["_cruise_label"] = (
                    df["CRUISE_ID"].map(cruise_labels).fillna(df["CRUISE_ID"].astype(str))
                )

            conn.close()
        except Exception as e:
            logger.exception("Scatter query failed")
            return no_update, no_update, *no_slider, str(e), True

        if df.empty:
            _scatter_cache["df"] = None
            return no_update, no_update, *no_slider, "No data matched the filters.", True

        # Parse dates for the year slider
        if "DATEANDTIME" in df.columns:
            df["_year"] = pd.to_datetime(
                df["DATEANDTIME"], errors="coerce",
            ).dt.year
        else:
            df["_year"] = np.nan

        # Populate server-side cache
        _scatter_cache["df"] = df
        _scatter_cache["bounds"] = {
            "lon_min": lon_min, "lon_max": lon_max,
            "lat_min": lat_min, "lat_max": lat_max,
            "param": param,
        }
        _scatter_cache["unit_note"] = " (units converted)" if converted else ""

        # Slider ranges from the actual data
        z_min = float(df["LEV_M"].min()) if "LEV_M" in df.columns else 0
        z_max = float(df["LEV_M"].max()) if "LEV_M" in df.columns else 6000
        y_min = int(df["_year"].min()) if df["_year"].notna().any() else 1900
        y_max = int(df["_year"].max()) if df["_year"].notna().any() else 2025

        # Render with full data
        unit_note = _scatter_cache["unit_note"]
        status = f"{len(df):,} rows loaded  -  {param}{unit_note}"
        fig = _scatter_figure(df, param, plot_type, _scatter_cache["bounds"], show_global, colour_by)
        return (fig, status,
                z_min, z_max, [z_min, z_max],
                y_min, y_max, [y_min, y_max],
                "", False)

    # Slider or plot-type change: filter cached data
    df = _scatter_cache.get("df")
    bounds = _scatter_cache.get("bounds")
    if df is None or bounds is None:
        raise PreventUpdate

    param = bounds["param"]
    filtered = df

    # Apply depth filter
    if depth_range and "LEV_M" in filtered.columns:
        filtered = filtered[
            (filtered["LEV_M"] >= depth_range[0])
            & (filtered["LEV_M"] <= depth_range[1])
            ]

    # Apply year filter
    if year_range and "_year" in filtered.columns:
        filtered = filtered[
            filtered["_year"].isna()
            | ((filtered["_year"] >= year_range[0])
               & (filtered["_year"] <= year_range[1]))
            ]

    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            annotations=[dict(text="No data in selected range",
                              showarrow=False, font_size=16)],
        )
        return fig, "0 rows after filtering", *no_slider, "", False

    unit_note = _scatter_cache.get("unit_note", "")
    status = f"{len(filtered):,} rows  -  {param}{unit_note}"
    fig = _scatter_figure(filtered, param, plot_type, bounds, show_global, colour_by)
    return fig, status, *no_slider, "", False


def _scatter_figure(df, param, plot_type, bounds, show_global=False, colour_by="VAL"):
    """Build a Plotly figure for the scatter tab."""
    # Crop map to the queried spatial extent with padding
    lat_min = bounds.get("lat_min", -90)
    lat_max = bounds.get("lat_max", 90)
    lon_min = bounds.get("lon_min", -180)
    lon_max = bounds.get("lon_max", 180)
    lat_pad = (lat_max - lat_min) * 0.05
    lon_pad = (lon_max - lon_min) * 0.05

    # Resolve the "Colour by" selection
    colour_col = colour_by if colour_by in df.columns else "VAL"
    colour_label = param if colour_col == "VAL" else _COLOUR_BY_LABELS.get(colour_col, colour_col)
    extra_colour = None if colour_col == "VAL" else colour_col

    if plot_type == "map" and "LATITUDE" in df.columns:
        # Aggregate to one value per lat/lon (mean)
        is_aggregated = df.duplicated(subset=["LATITUDE", "LONGITUDE"]).any()
        agg_kwargs = {colour_col: (colour_col, "mean"), "n_obs": (colour_col, "size")}
        if "_station_label" in df.columns:
            agg_kwargs["stations"] = ("_station_label", _format_hover_list)
        if "_cruise_label" in df.columns:
            agg_kwargs["cruises"] = ("_cruise_label", _format_hover_list)
        plot_df = df.groupby(["LATITUDE", "LONGITUDE"], as_index=False).agg(**agg_kwargs)

        title_suffix = " (mean)" if is_aggregated else ""
        hover_data = {"n_obs": True}
        hover_labels = {colour_col: colour_label, "n_obs": "# obs"}
        for extra_col, label in (("stations", "Stations"), ("cruises", "Cruises")):
            if extra_col in plot_df.columns:
                hover_data[extra_col] = True
                hover_labels[extra_col] = label

        fig = px.scatter_geo(
            plot_df, lat="LATITUDE", lon="LONGITUDE", color=colour_col,
            color_continuous_scale="Viridis",
            labels=hover_labels,
            hover_data=hover_data,
            title=f"{colour_label}{title_suffix} - spatial distribution",
        )
        if show_global:
            fig.update_geos(
                **_GEO_STYLE,
                lataxis_range=[-90, 90],
                lonaxis_range=[-180, 180],
            )
        else:
            fig.update_geos(
                **_GEO_STYLE,
                lataxis_range=[lat_min - lat_pad, lat_max + lat_pad],
                lonaxis_range=[lon_min - lon_pad, lon_max + lon_pad],
            )
    elif plot_type == "profile":
        fig = px.scatter(
            df, x="VAL", y="LEV_M", color=extra_colour,
            labels={"VAL": param, "LEV_M": "Depth (m)", colour_col: colour_label},
            title=f"{param} - depth profile" + (f", coloured by {colour_label}" if extra_colour else ""),
            opacity=0.3 if extra_colour is None else 0.6,
        )
        fig.update_yaxes(autorange="reversed")
    elif plot_type == "time" and "DATEANDTIME" in df.columns:
        temp = df.copy()
        temp["DATEANDTIME"] = pd.to_datetime(
            temp["DATEANDTIME"], errors="coerce",
        )
        fig = px.scatter(
            temp, x="DATEANDTIME", y="VAL", color=extra_colour,
            labels={"VAL": param, "DATEANDTIME": "Date", colour_col: colour_label},
            title=f"{param} - time series" + (f", coloured by {colour_label}" if extra_colour else ""),
            opacity=0.3 if extra_colour is None else 0.6,
        )
    else:
        fig = px.histogram(
            df, x="VAL", color=extra_colour, nbins=80,
            labels={"VAL": param, colour_col: colour_label},
            title=f"{param} - value distribution" + (f", coloured by {colour_label}" if extra_colour else ""),
        )

    fig.update_layout(template="plotly_white")
    return fig


# --- Grid callbacks ----------------------------------------------------


# Refresh grid table dropdown
@_register_callback(
    Output("grid-table", "options"),
    Output("grid-table", "value"),
    Input("tabs", "active_tab"),
    Input("btn-refresh-grid", "n_clicks"),
)
def _refresh_wide_tables(_tab, _n):
    """Populate the wide table dropdown with available grid tables."""
    tables = _query_wide_tables()
    options = [{"label": t, "value": t} for t in tables]
    return options, None


# Configure grid sliders when a wide table is selected
@_register_callback(
    Output("grid-param", "options"),
    Output("grid-param", "value"),
    Output("grid-depth-slider", "min"),
    Output("grid-depth-slider", "max"),
    Output("grid-depth-slider", "value"),
    Output("grid-depth-slider", "marks"),
    Output("grid-depth-slider", "step"),
    Output("grid-depth-slider", "disabled"),
    Output("grid-time-slider", "min"),
    Output("grid-time-slider", "max"),
    Output("grid-time-slider", "value"),
    Output("grid-time-slider", "marks"),
    Output("grid-time-slider", "step"),
    Output("grid-time-slider", "disabled"),
    Input("grid-table", "value"),
)
def _populate_grid_controls(table_name):
    """Read the wide table structure and configure depth and time range sliders."""
    if not table_name:
        return (
            [], None,
            0, 1, [0, 1], {0: "-"}, None, True,
            0, 1, [0, 1], {0: "-"}, None, True,
        )

    conn = _get_conn()
    try:
        # Read column names and distinct depth/time values
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        all_cols = [row[1] for row in cur.fetchall()]

        depths = [row[0] for row in conn.execute(
            f"SELECT DISTINCT LEV_M FROM {table_name} ORDER BY LEV_M"
        ).fetchall() if row[0] is not None]

        times = [row[0] for row in conn.execute(
            f"SELECT DISTINCT DATEANDTIME FROM {table_name} "
            f"ORDER BY DATEANDTIME"
        ).fetchall() if row[0] is not None]
    finally:
        conn.close()

    # Cache for the render callback
    _grid_cache["depths"] = [float(d) for d in depths]
    _grid_cache["times"] = [str(t) for t in times]

    # Identify parameter columns
    meta_cols = {"idx", "LATITUDE", "LONGITUDE", "LEV_M",
                 "DATEANDTIME", "water"}
    param_cols = [c for c in all_cols if c not in meta_cols]
    p_opts = [{"label": c, "value": c} for c in param_cols]

    # Depth range slider: Tick at every level, label only first and last
    if len(depths) > 1:
        d_marks = _make_slider_marks(
            [float(d) for d in depths],
            f"{float(depths[0]):.0f}m", f"{float(depths[-1]):.0f}m",
        )
        d_min, d_max = float(depths[0]), float(depths[-1])
        d_val = [d_min, d_min]
        d_step, d_disabled = None, False
    elif len(depths) == 1:
        d_marks = _make_slider_marks(
            [float(depths[0])],
            f"{float(depths[0]):.0f}m", f"{float(depths[0]):.0f}m",
        )
        d_min = d_max = float(depths[0])
        d_val = [d_min, d_min]
        d_step, d_disabled = None, True
    else:
        d_marks, d_min, d_max = {0: "-"}, 0, 1
        d_val = [0, 1]
        d_step, d_disabled = None, True

    # Time range slider: Index-based, snap to marks only
    if len(times) > 1:
        t_marks = _make_slider_marks(
            list(range(len(times))),
            str(times[0])[:10], str(times[-1])[:10],
        )
        t_min, t_max = 0, len(times) - 1
        t_val = [0, 0]
        t_step, t_disabled = None, False
    elif len(times) == 1:
        t_marks = {0: str(times[0])[:10]}
        t_min = t_max = 0
        t_val = [0, 0]
        t_step, t_disabled = None, True
    else:
        t_marks, t_min, t_max = {0: "-"}, 0, 1
        t_val = [0, 1]
        t_step, t_disabled = None, True

    return (
        p_opts, param_cols[0] if param_cols else None,
        d_min, d_max, d_val, d_marks, d_step, d_disabled,
        t_min, t_max, t_val, t_marks, t_step, t_disabled,
    )


# Render grid chart
@_register_callback(
    Output("chart", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("alert", "children", allow_duplicate=True),
    Output("alert", "is_open", allow_duplicate=True),
    Input("grid-param", "value"),
    Input("grid-depth-slider", "value"),
    Input("grid-time-slider", "value"),
    Input("grid-plot-type", "value"),
    Input("grid-global-map", "value"),
    State("grid-table", "value"),
    prevent_initial_call=True,
)
def _render_grid(value_col, depth_range, time_idx_range, plot_type, show_global, table_name):
    """Query a depth/time range from the wide table and render."""
    if not table_name or not value_col:
        raise PreventUpdate

    depths = _grid_cache.get("depths", [])
    times = _grid_cache.get("times", [])

    # Resolve time slider index range to actual time values
    time_range = None
    if times and time_idx_range is not None:
        lo, hi = int(time_idx_range[0]), int(time_idx_range[1])
        time_range = (times[lo], times[hi])

    # Query only the selected depth/time range
    try:
        conn = _get_conn()
        conditions = []
        params = []
        if depth_range is not None and depths:
            conditions.append("LEV_M >= ? AND LEV_M <= ?")
            params.extend([float(depth_range[0]), float(depth_range[1])])
        if time_range is not None:
            conditions.append("DATEANDTIME >= ? AND DATEANDTIME <= ?")
            params.extend([time_range[0], time_range[1]])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        df = pd.read_sql_query(
            f"SELECT * FROM {table_name} {where}", conn, params=params,
        )
        conn.close()
    except Exception as e:
        logger.exception("Grid query failed")
        return no_update, no_update, str(e), True

    if df.empty:
        return no_update, "0 rows", "No data for this selection.", True

    # Build status text
    n_total = len(df)
    n_valid = df[value_col].notna().sum()
    depth_label = (
        _format_range_label(depth_range[0], depth_range[1], "{:.0f}") + " m"
        if depth_range is not None else "all"
    )
    time_label = (
        _format_range_label(str(time_range[0])[:10], str(time_range[1])[:10])
        if time_range is not None else "all"
    )
    status = (f"{n_valid:,} / {n_total:,} cells - "
              f"{value_col}  |  depth: {depth_label}  |  time: {time_label}")

    label = f"depth {depth_label}" + (f", time {time_label}" if time_range is not None else "")
    fig = _grid_figure(df, value_col, label, plot_type, show_global)
    return fig, status, "", False


@_register_callback(
    Output("grid-depth-display", "children"),
    Input("grid-depth-slider", "value"),
)
def _show_grid_depth(val):
    if not val:
        return ""
    return f"Depth: {_format_range_label(val[0], val[1], '{:.0f}')}m"


@_register_callback(
    Output("grid-time-display", "children"),
    Input("grid-time-slider", "value"),
)
def _show_grid_time(idx_range):
    times = _grid_cache.get("times", [])
    if not times or not idx_range:
        return ""
    try:
        lo, hi = int(idx_range[0]), int(idx_range[1])
        return f"Datetime: {_format_range_label(str(times[lo]), str(times[hi]))}"
    except Exception:
        return ""


def _grid_figure(df, value_col, depth_label, plot_type, show_global=False):
    """Build a Plotly figure for the grid tab."""
    if plot_type == "map":
        # Show only cells with data, aggregated to one value per lat/lon (mean)
        valid = df[df[value_col].notna()]
        is_aggregated = valid.duplicated(subset=["LATITUDE", "LONGITUDE"]).any()
        plot_df = valid.groupby(["LATITUDE", "LONGITUDE"], as_index=False)[value_col].mean()
        title_suffix = " (mean)" if is_aggregated else ""
        fig = px.scatter_geo(
            plot_df, lat="LATITUDE", lon="LONGITUDE", color=value_col,
            color_continuous_scale="Viridis",
            title=f"{value_col}{title_suffix} - {depth_label}",
        )
        # Crop map to grid extent
        lat_min, lat_max = df["LATITUDE"].min(), df["LATITUDE"].max()
        lon_min, lon_max = df["LONGITUDE"].min(), df["LONGITUDE"].max()
        lat_pad = (lat_max - lat_min) * 0.05
        lon_pad = (lon_max - lon_min) * 0.05
        if show_global:
            fig.update_geos(
                **_GEO_STYLE,
                lataxis_range=[-90, 90],
                lonaxis_range=[-180, 180],
            )
        else:
            fig.update_geos(
                **_GEO_STYLE,
                lataxis_range=[lat_min - lat_pad, lat_max + lat_pad],
                lonaxis_range=[lon_min - lon_pad, lon_max + lon_pad],
            )
    elif plot_type == "missing":
        # Missing value fraction per parameter column
        meta_cols = {"idx", "LATITUDE", "LONGITUDE", "LEV_M",
                     "DATEANDTIME", "water"}
        param_cols = [c for c in df.columns if c not in meta_cols]
        missing = pd.DataFrame({
            "parameter": param_cols,
            "missing_pct": [
                (df[c].isna().sum() / len(df)) * 100 for c in param_cols
            ],
        }).sort_values("missing_pct")
        fig = px.bar(
            missing, x="parameter", y="missing_pct",
            labels={"missing_pct": "Missing (%)", "parameter": ""},
            color="missing_pct", color_continuous_scale="RdYlGn_r",
            title=f"Missing values - {depth_label}",
            text=missing["missing_pct"].round(1).astype(str) + "%",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
    else:
        # Histogram of non-null values
        valid = df[value_col].dropna()
        fig = px.histogram(
            valid, x=value_col, nbins=80,
            title=f"{value_col} - {depth_label}",
        )

    fig.update_layout(template="plotly_white")
    return fig


# --- CSV callbacks ----------------------------------------------------


@_register_callback(
    Output("csv-filename", "children"),
    Output("csv-param", "options"),
    Output("csv-param", "value"),
    Output("csv-depth-slider", "min"),
    Output("csv-depth-slider", "max"),
    Output("csv-depth-slider", "value"),
    Output("csv-depth-slider", "marks"),
    Output("csv-depth-slider", "step"),
    Output("csv-depth-slider", "disabled"),
    Output("csv-time-slider", "min"),
    Output("csv-time-slider", "max"),
    Output("csv-time-slider", "value"),
    Output("csv-time-slider", "marks"),
    Output("csv-time-slider", "step"),
    Output("csv-time-slider", "disabled"),
    Input("csv-upload", "contents"),
    State("csv-upload", "filename"),
    prevent_initial_call=True,
)
def _load_csv(contents, filename):
    if contents is None:
        raise PreventUpdate

    # Read CSV
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    # DATEANDTIME is optional
    has_time = "DATEANDTIME" in df.columns
    if has_time:
        df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"])
    _csv_cache["df"] = df

    # Available depth/time values
    depths = np.sort(df["LEV_M"].dropna().unique())
    times = np.sort(df["DATEANDTIME"].unique()) if has_time else np.array([])

    _csv_cache["depths"] = [float(d) for d in depths]
    _csv_cache["times"] = list(times)

    # Parameter columns
    meta_cols = {
        "idx", "LATITUDE", "LONGITUDE", "LEV_M",
        "DATEANDTIME", "water", "miss",
    }
    param_cols = [c for c in df.columns if c not in meta_cols]
    p_opts = [{"label": c, "value": c} for c in param_cols]

    # Depth range slider: Tick at every level, label only first and last
    if len(depths) > 1:
        d_marks = _make_slider_marks(
            [float(d) for d in depths],
            f"{float(depths[0]):.0f}m", f"{float(depths[-1]):.0f}m",
        )
        d_min, d_max = float(depths[0]), float(depths[-1])
        d_val = [d_min, d_min]
        d_step, d_disabled = None, False
    elif len(depths) == 1:
        d_marks = _make_slider_marks(
            [float(depths[0])],
            f"{float(depths[0]):.0f}m", f"{float(depths[0]):.0f}m",
        )
        d_min = d_max = float(depths[0])
        d_val = [d_min, d_min]
        d_step, d_disabled = None, True
    else:
        d_marks, d_min, d_max = {0: "-"}, 0, 1
        d_val = [0, 1]
        d_step, d_disabled = None, True

    # Time range slider: Index-based, snap to marks only
    if len(times) > 1:
        t_marks = _make_slider_marks(
            list(range(len(times))),
            str(pd.Timestamp(times[0]))[:10], str(pd.Timestamp(times[-1]))[:10],
        )
        t_min, t_max = 0, len(times) - 1
        t_val = [0, 0]
        t_step, t_disabled = None, False
    elif len(times) == 1:
        t_marks = {0: str(pd.Timestamp(times[0]))[:10]}
        t_min = t_max = 0
        t_val = [0, 0]
        t_step, t_disabled = None, True
    else:
        t_marks, t_min, t_max = {0: "-"}, 0, 1
        t_val = [0, 1]
        t_step, t_disabled = None, True

    return (
        f"Loaded: {filename} ({len(df):,} rows)",
        p_opts, param_cols[0] if param_cols else None,
        d_min, d_max, d_val, d_marks, d_step, d_disabled,
        t_min, t_max, t_val, t_marks, t_step, t_disabled,
    )


# Render CSV chart
@_register_callback(
    Output("chart", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("alert", "children", allow_duplicate=True),
    Output("alert", "is_open", allow_duplicate=True),
    Input("csv-param", "value"),
    Input("csv-depth-slider", "value"),
    Input("csv-time-slider", "value"),
    Input("csv-plot-type", "value"),
    Input("csv-global-map", "value"),
    prevent_initial_call=True,
)
def _render_csv(value_col, depth_range, time_idx_range, plot_type, show_global):
    """Query a depth/time range from the CSV table and render."""
    if _csv_cache["df"] is None or not value_col:
        raise PreventUpdate

    times = _csv_cache.get("times", [])

    # Resolve time slider index range to actual time values
    time_range = None
    if times and time_idx_range is not None:
        lo, hi = int(time_idx_range[0]), int(time_idx_range[1])
        time_range = (times[lo], times[hi])

    # Get the selected depth/time range (DATEANDTIME may be absent)
    df = _csv_cache["df"]
    mask = df["LEV_M"].between(depth_range[0], depth_range[1]) if depth_range is not None else True
    if "DATEANDTIME" in df.columns and time_range is not None:
        mask &= df["DATEANDTIME"].between(time_range[0], time_range[1])
    df = df[mask]

    if df.empty:
        return no_update, "0 rows", "No data for this selection.", True

    # Build status text
    n_total = len(df)
    n_valid = df[value_col].notna().sum()
    depth_label = (
        _format_range_label(depth_range[0], depth_range[1], "{:.0f}") + " m"
        if depth_range is not None else "all"
    )
    time_label = (
        _format_range_label(str(time_range[0])[:10], str(time_range[1])[:10])
        if time_range is not None else "all"
    )
    status = (f"{n_valid:,} / {n_total:,} cells - "
              f"{value_col} | depth: {depth_label} | time: {time_label}")

    label = f"depth {depth_label}" + (f", time {time_label}" if time_range is not None else "")
    fig = _grid_figure(df, value_col, label, plot_type, show_global)
    return fig, status, "", False


@_register_callback(
    Output("csv-depth-display", "children"),
    Input("csv-depth-slider", "value"),
)
def _show_csv_depth(val):
    if not val:
        return ""
    return f"Depth: {_format_range_label(val[0], val[1], '{:.0f}')}m"


@_register_callback(
    Output("csv-time-display", "children"),
    Input("csv-time-slider", "value"),
)
def _show_csv_time(idx_range):
    # Use csv cache, not grid cache
    times = _csv_cache.get("times", [])
    if not times or not idx_range:
        return ""
    try:
        lo, hi = int(idx_range[0]), int(idx_range[1])
        lo_s = pd.Timestamp(times[lo]).strftime("%Y-%m-%d %H:%M:%S")
        hi_s = pd.Timestamp(times[hi]).strftime("%Y-%m-%d %H:%M:%S")
        return f"Datetime: {_format_range_label(lo_s, hi_s)}"
    except Exception:
        return ""


def create_app(db_path: str | os.PathLike | None = None) -> Dash:
    """Create the COMFORT dataset Explorer app.

    Args:
        db_path (str or Path): Path to the COMFORT SQLite database. ``None``
            reads the ``COMFORT_DB_PATH`` environment variable (a ``.env``
            file is loaded if python-dotenv is installed).
    Returns:
        dash.Dash: The app.
    Raises:
        EnvironmentError: If no database path is given and ``COMFORT_DB_PATH`` is not set.
    """
    global _db_path

    # Pick up COMFORT_DB_PATH from a .env file if available
    if load_dotenv is not None:
        load_dotenv()

    tmp_path = db_path or os.environ.get("COMFORT_DB_PATH")
    if not tmp_path:
        raise EnvironmentError(
            "No database path given and COMFORT_DB_PATH environment variable is not set.\n"
            "    export COMFORT_DB_PATH=/path/to/comfort.sqlite"
        )
    _db_path = str(tmp_path)

    # Build app
    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="COMFORT Explorer")
    app.layout = _build_layout(_query_parameters())

    # Register callbacks
    for args, kwargs, func in _CALLBACKS:
        app.callback(*args, **kwargs)(func)
    return app


def main() -> None:
    """Launch the app in debug mode (``python -m comfort.app.explorer``)."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    create_app().run(debug=True)


if __name__ == "__main__":
    main()
