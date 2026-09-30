import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_FILE = "/media/galdrux/galdrux_storage/sem_5/DS/dataset/battery_features.csv"
TARGET = "capacity"

FEATURES = [
    "cycle",
    "mean_voltage",
    "min_voltage",
    "max_voltage",
    "mean_current",
    "mean_temperature",
    "max_temperature",
    "discharge_time"
]

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"\nCSV file '{CSV_FILE}' was not found.\n"
        f"Put your CSV file in the same folder as dashboard.py."
    )

df = pd.read_csv(CSV_FILE)

df.columns = df.columns.str.strip()

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df = df[required_columns].copy()

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

if len(df) < 10:
    raise ValueError("The dataset contains too few valid rows.")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

filter_selector = SelectKBest(
    score_func=f_regression,
    k=4
)

filter_selector.fit(
    X_train,
    y_train
)

filter_features = X_train.columns[
    filter_selector.get_support()
].tolist()

filter_scores = pd.Series(
    filter_selector.scores_,
    index=X_train.columns
).sort_values(
    ascending=False
)

rfe = RFE(
    estimator=LinearRegression(),
    n_features_to_select=4
)

rfe.fit(
    X_train,
    y_train
)

rfe_features = X_train.columns[
    rfe.support_
].tolist()

rfe_ranking = pd.Series(
    rfe.ranking_,
    index=X_train.columns
).sort_values()

rf_feature_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_feature_model.fit(
    X_train,
    y_train
)

tree_importance = pd.Series(
    rf_feature_model.feature_importances_,
    index=X_train.columns
).sort_values(
    ascending=False
)

tree_features = tree_importance.head(4).index.tolist()

feature_sets = {
    "All Features": FEATURES,
    "Filter Method": filter_features,
    "RFE Method": rfe_features,
    "Tree-Based Method": tree_features
}

model_results = []

for method, selected_features in feature_sets.items():

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.perf_counter()

    model.fit(
        X_train[selected_features],
        y_train
    )

    training_time = time.perf_counter() - start_time

    predictions = model.predict(
        X_test[selected_features]
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    model_results.append({
        "Method": method,
        "Features": len(selected_features),
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "Training Time (s)": training_time
    })

results_df = pd.DataFrame(model_results)

correlation = (
    df.corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .sort_values(
        key=abs,
        ascending=False
    )
)

avg_capacity = df[TARGET].mean()
min_capacity = df[TARGET].min()
max_capacity = df[TARGET].max()

cycle_min = int(df["cycle"].min())
cycle_max = int(df["cycle"].max())

missing_values = int(
    df.isnull().sum().sum()
)

app = Dash(__name__)

app.title = "Battery Feature Selection Dashboard"

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "12px",
    "padding": "18px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
    "textAlign": "center",
    "flex": "1",
    "minWidth": "180px"
}

SECTION_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "12px",
    "padding": "20px",
    "marginTop": "20px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"
}

app.layout = html.Div(
    style={
        "backgroundColor": "#f4f6f8",
        "minHeight": "100vh",
        "padding": "25px",
        "fontFamily": "Arial"
    },
    children=[

        html.Div(
            style={
                "backgroundColor": "#1f2937",
                "color": "white",
                "padding": "25px",
                "borderRadius": "12px",
                "marginBottom": "20px"
            },
            children=[

                html.H1(
                    "Battery Feature Selection Dashboard",
                    style={
                        "margin": "0",
                        "fontSize": "32px"
                    }
                ),

                html.P(
                    "Interactive analysis of battery-system data "
                    "using statistical and machine-learning "
                    "feature selection techniques.",
                    style={
                        "marginBottom": "0",
                        "fontSize": "16px"
                    }
                )
            ]
        ),

        html.Div(
            style={
                "display": "flex",
                "gap": "15px",
                "flexWrap": "wrap"
            },
            children=[

                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.H4("Observations"),
                        html.H2(f"{len(df)}")
                    ]
                ),

                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.H4("Features"),
                        html.H2(f"{len(FEATURES)}")
                    ]
                ),

                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.H4("Average Capacity"),
                        html.H2(f"{avg_capacity:.4f}")
                    ]
                ),

                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.H4("Minimum Capacity"),
                        html.H2(f"{min_capacity:.4f}")
                    ]
                ),

                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.H4("Maximum Capacity"),
                        html.H2(f"{max_capacity:.4f}")
                    ]
                ),

                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.H4("Missing Values"),
                        html.H2(f"{missing_values}")
                    ]
                )
            ]
        ),

        html.Div(
            style=SECTION_STYLE,
            children=[

                html.H2(
                    "Battery Capacity Analysis"
                ),

                html.Label(
                    "Select Cycle Range",
                    style={
                        "fontWeight": "bold"
                    }
                ),

                dcc.RangeSlider(
                    id="cycle-slider",
                    min=cycle_min,
                    max=cycle_max,
                    value=[
                        cycle_min,
                        cycle_max
                    ],
                    marks={
                        cycle_min: str(cycle_min),
                        cycle_max: str(cycle_max)
                    },
                    allowCross=False,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True
                    }
                ),

                dcc.Graph(
                    id="capacity-cycle-graph",
                    style={
                        "height": "500px"
                    }
                )
            ]
        ),

        html.Div(
            style=SECTION_STYLE,
            children=[

                html.H2(
                    "Interactive Feature Analysis"
                ),

                html.Label(
                    "Select Feature",
                    style={
                        "fontWeight": "bold"
                    }
                ),

                dcc.Dropdown(
                    id="feature-dropdown",
                    options=[
                        {
                            "label": feature.replace(
                                "_",
                                " "
                            ).title(),
                            "value": feature
                        }
                        for feature in FEATURES
                    ],
                    value="mean_voltage",
                    clearable=False
                ),

                dcc.Graph(
                    id="feature-capacity-graph",
                    style={
                        "height": "500px"
                    }
                )
            ]
        ),

        html.Div(
            style=SECTION_STYLE,
            children=[

                html.H2(
                    "Correlation Analysis"
                ),

                dcc.Graph(
                    id="correlation-heatmap",
                    figure=px.imshow(
                        df.corr(
                            numeric_only=True
                        ),
                        text_auto=".2f",
                        aspect="auto",
                        title="Correlation Heatmap"
                    )
                ),

                dcc.Graph(
                    id="correlation-bar",
                    figure=px.bar(
                        x=correlation.index,
                        y=correlation.values,
                        title="Feature Correlation with Capacity",
                        labels={
                            "x": "Feature",
                            "y": "Correlation"
                        }
                    )
                )
            ]
        ),

        html.Div(
            style=SECTION_STYLE,
            children=[

                html.H2(
                    "Feature Selection Results"
                ),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns":
                            "repeat(auto-fit,minmax(250px,1fr))",
                        "gap": "15px"
                    },
                    children=[

                        html.Div(
                            style={
                                "padding": "15px",
                                "borderRadius": "10px",
                                "backgroundColor": "#eef2ff"
                            },
                            children=[

                                html.H3(
                                    "Filter Method"
                                ),

                                html.Ul([
                                    html.Li(
                                        feature
                                    )
                                    for feature
                                    in filter_features
                                ])
                            ]
                        ),

                        html.Div(
                            style={
                                "padding": "15px",
                                "borderRadius": "10px",
                                "backgroundColor": "#ecfeff"
                            },
                            children=[

                                html.H3(
                                    "RFE Method"
                                ),

                                html.Ul([
                                    html.Li(
                                        feature
                                    )
                                    for feature
                                    in rfe_features
                                ])
                            ]
                        ),

                        html.Div(
                            style={
                                "padding": "15px",
                                "borderRadius": "10px",
                                "backgroundColor": "#f0fdf4"
                            },
                            children=[

                                html.H3(
                                    "Tree-Based Method"
                                ),

                                html.Ul([
                                    html.Li(
                                        feature
                                    )
                                    for feature
                                    in tree_features
                                ])
                            ]
                        )
                    ]
                ),

                dcc.Graph(
                    id="filter-score-graph",
                    figure=px.bar(
                        x=filter_scores.index,
                        y=filter_scores.values,
                        title="Filter Method - F Scores",
                        labels={
                            "x": "Feature",
                            "y": "F Score"
                        }
                    )
                ),

                dcc.Graph(
                    id="tree-importance-graph",
                    figure=px.bar(
                        x=tree_importance.index,
                        y=tree_importance.values,
                        title="Random Forest Feature Importance",
                        labels={
                            "x": "Feature",
                            "y": "Importance"
                        }
                    )
                )
            ]
        ),

        html.Div(
            style=SECTION_STYLE,
            children=[

                html.H2(
                    "Model Performance Comparison"
                ),

                dcc.Graph(
                    id="r2-graph",
                    figure=px.bar(
                        results_df,
                        x="Method",
                        y="R2 Score",
                        title="R² Score Comparison",
                        text_auto=".4f"
                    )
                ),

                dcc.Graph(
                    id="rmse-graph",
                    figure=px.bar(
                        results_df,
                        x="Method",
                        y="RMSE",
                        title="RMSE Comparison",
                        text_auto=".4f"
                    )
                ),

                dash_table.DataTable(
                    id="results-table",
                    data=results_df.round(
                        6
                    ).to_dict(
                        "records"
                    ),
                    columns=[
                        {
                            "name": col,
                            "id": col
                        }
                        for col
                        in results_df.columns
                    ],
                    sort_action="native",
                    filter_action="native",
                    style_table={
                        "overflowX": "auto"
                    },
                    style_cell={
                        "textAlign": "center",
                        "padding": "10px"
                    },
                    style_header={
                        "fontWeight": "bold"
                    }
                )
            ]
        ),

        html.Div(
            style=SECTION_STYLE,
            children=[

                html.H2(
                    "Dataset"
                ),

                dash_table.DataTable(
                    id="data-table",
                    data=df.round(
                        6
                    ).to_dict(
                        "records"
                    ),
                    columns=[
                        {
                            "name": col,
                            "id": col
                        }
                        for col
                        in df.columns
                    ],
                    page_size=15,
                    sort_action="native",
                    filter_action="native",
                    page_action="native",
                    style_table={
                        "overflowX": "auto"
                    },
                    style_cell={
                        "textAlign": "center",
                        "padding": "8px",
                        "minWidth": "120px"
                    },
                    style_header={
                        "fontWeight": "bold"
                    }
                )
            ]
        ),

        html.Div(
            style={
                "textAlign": "center",
                "marginTop": "25px",
                "color": "#666"
            },
            children=[

                html.P(
                    "Experiment 03 - Feature Selection "
                    "for Autonomous Systems"
                )
            ]
        )
    ]
)


@app.callback(
    Output(
        "capacity-cycle-graph",
        "figure"
    ),
    Input(
        "cycle-slider",
        "value"
    )
)
def update_capacity_cycle_graph(
    cycle_range
):

    if (
        cycle_range is None
        or len(cycle_range) != 2
    ):
        cycle_range = [
            cycle_min,
            cycle_max
        ]

    filtered = df[
        (df["cycle"] >= cycle_range[0])
        &
        (df["cycle"] <= cycle_range[1])
    ].copy()

    if filtered.empty:

        fig = go.Figure()

        fig.add_annotation(
            text="No data available for this cycle range.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False
        )

        fig.update_layout(
            title="Battery Capacity vs Cycle"
        )

        return fig

    fig = px.scatter(
        filtered,
        x="cycle",
        y="capacity",
        title="Battery Capacity vs Cycle",
        labels={
            "cycle": "Cycle",
            "capacity": "Capacity"
        }
    )

    fig.update_traces(
        marker={
            "size": 9
        }
    )

    fig.update_layout(
        hovermode="closest",
        template="plotly_white"
    )

    return fig


@app.callback(
    Output(
        "feature-capacity-graph",
        "figure"
    ),
    Input(
        "feature-dropdown",
        "value"
    )
)
def update_feature_graph(
    feature
):

    if (
        feature is None
        or feature not in FEATURES
    ):
        feature = FEATURES[0]

    plot_df = df[
        [feature, TARGET]
    ].dropna().copy()

    if plot_df.empty:

        fig = go.Figure()

        fig.add_annotation(
            text="No data available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False
        )

        return fig

    fig = px.scatter(
        plot_df,
        x=feature,
        y=TARGET,
        title=(
            f"{feature.replace('_', ' ').title()} "
            f"vs Capacity"
        ),
        labels={
            feature:
                feature.replace(
                    "_",
                    " "
                ).title(),
            TARGET: "Capacity"
        }
    )

    fig.update_traces(
        marker={
            "size": 9
        }
    )

    fig.update_layout(
        hovermode="closest",
        template="plotly_white"
    )

    return fig


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print(
        "BATTERY FEATURE SELECTION DASHBOARD"
    )
    print("=" * 60)

    print(
        f"Dataset: {CSV_FILE}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"Target: {TARGET}"
    )

    print("\nSelected Features:")

    print(
        "Filter:",
        filter_features
    )

    print(
        "RFE:",
        rfe_features
    )

    print(
        "Tree-Based:",
        tree_features
    )

    print("\nStarting dashboard...")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050
    )