import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html

import copy
import requests
import time
import numpy as np
import pandas as pd

def get_predictions(payment_history, number_queries):
    predictions = pd.DataFrame(columns=["prediction", "model", "time"])

    if payment_history == "on-time":
        payment_months = [-1, -1, -1]
    elif payment_history == "2-4-months-late":
        payment_months = [2, 3, 4]
    elif payment_history == "7-9-months-late":
        payment_months = [7, 8, 9]

    for i in range(number_queries):
        model_endpoint = "https://colorado.rstudio.com/rsc/model-management-python/model-router/predict"
        months = str(np.random.choice(payment_months, size=1)[0])
        params = "35,500000,1,1,1,58," + (months + ",") * 6 + "13709,5006,31130,3180,0,5293,5006,31178,3180,0,5293,768"
        post_data = {"input": params}

        t0 = time.time()
        r = requests.post(model_endpoint, json=post_data)
        t1 = time.time()
        time_diff = t1 - t0

        resp = r.json()
        predictions = predictions.append({"prediction": resp["prediction"],
                                            "model": resp["model"],
                                            "time": time_diff},
                                            ignore_index=True)

    return predictions

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server

app.title = "Model Predictions Dashboard"

app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("rstudio-logo.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Model Predictions Dashboard",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Powered by Python and RStudio Connect", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Learn More", id="learn-more-button"),
                            href="https://solutions.rstudio.com/model-management/overview/",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Payment History", style={"font-weight": "bold"}),
                        html.Br(),
                        dcc.RadioItems(
                            id="payment_history",
                            options=[
                                {"label": " On Time Payments ", "value": "on-time"},
                                {"label": " 2 to 4 Months Late ", "value": "2-4-months-late"},
                                {"label": " 7 to 9 Months Late ", "value": "7-9-months-late"},
                            ],
                            value="2-4-months-late",
                            className="dcc_control",
                        ),
                        html.Br(),
                        html.Br(),
                        html.Label("Number of Model Queries", style={"font-weight": "bold"}),
                        html.Br(),
                        dcc.Slider(
                            id="number_queries",
                            min=10,
                            max=50,
                            marks={i: "Label {}".format(i) if i == 1 else str(i) for i in range(10, 51, 10)},
                            value=10,
                        ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="number_api_queries"), html.P("Number of API Queries")],
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="model_latency"), html.P("Average Round Trip Time")],
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="model_a_average"), html.P("Model A Average Probability")],
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="model_b_average"), html.P("Model B Average Probability")],
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="predictions_graph"),
                                dcc.Interval(
                                    id="interval-component",
                                    interval=10*1000,
                                    n_intervals=0
                                )
                            ],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("predictions_graph", "figure")],
)

@app.callback(
    [
        Output("predictions_graph", "figure"),
        Output("number_api_queries", "children"),
        Output("model_latency", "children"),
        Output("model_a_average", "children"),
        Output("model_b_average", "children"),
    ],
    [
        Input("payment_history", "value"),
        Input("number_queries", "value"),
        Input("aggregate_data", "data"),
        Input("interval-component", "n_intervals"),
    ],
)
def make_count_figure(payment_history, number_queries, aggregate_data, n_intervals):

    layout = dict(
        autosize=True,
        automargin=True,
        margin=dict(l=50, r=30, b=40, t=40),
        hovermode="closest",
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        legend=dict(font=dict(size=10), orientation="h"),
    )

    df = get_predictions(payment_history, number_queries)

    time_average = df["time"].mean() * 1000
    model_a_average = np.round(df[df["model"] == "model-a"]["prediction"].astype(float).mean(), 2)
    model_b_average = np.round(df[df["model"] == "model-b"]["prediction"].astype(float).mean(), 2)

    time_average = str("{:3.0f}".format(time_average))

    if np.isnan(model_a_average):
        model_a_average = "-"
    else:
        model_a_average = str("{:0.2f}".format(model_a_average))
    if np.isnan(model_b_average):
        model_b_average = "-"
    else:
        model_b_average = str("{:0.2f}".format(model_b_average))

    data = [
        dict(
            type="bar",
            x=df[df["model"] == "model-a"].index,
            y=df[df["model"] == "model-a"]["prediction"].astype(float),
            name="Model A",
            marker=dict(color="skyblue"),
        ),
        dict(
            type="bar",
            x=df[df["model"] == "model-b"].index,
            y=df[df["model"] == "model-b"]["prediction"].astype(float),
            name="Model B",
            marker=dict(color="steelblue"),
        ),
    ]

    layout["title"] = "Model Predictions"
    layout["dragmode"] = "select"
    layout["showlegend"] = True
    layout["autosize"] = True
    layout["xaxis"] = dict(title="API Request Number")
    layout["yaxis"] = dict(title="Probability of Default", range=[0,1])

    figure = dict(data=data, layout=layout)
    return figure, number_queries, time_average + " ms", model_a_average, model_b_average

if __name__ == "__main__":
    app.run_server(debug=True)
