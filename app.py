import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import os

import pandas as pd
import numpy as np

from sustainable-deepfashion-dash.src.inference import distance, ModelInference

import plotly.graph_objects as go


basename = "simple_conv2d_embedding_size_16_angular_d-0"
model = ModelInference("tf_saved_model/")
prediction_csv_path = os.path.join(f"{basename}_predictions.csv")
prediction_df = pd.read_csv(prediction_csv_path)
prediction_df["prediction"] = prediction_df["prediction"].apply(lambda x: np.array(eval(x)).flatten())

WEB_IMAGES = True

NUMBER_OF_BEST_PREDICTIONS = 6

NUMBER_OF_PCA_SLIDERS = 3
# pca = PCA(n_components=NUMBER_OF_PCA_SLIDERS)
# pca.fit(prediction_df["prediction"].tolist())
# pca_components = pca.components_

EMBEDDING_SIZE = 16

slider_input = []
for i in range(NUMBER_OF_PCA_SLIDERS):
    slider_input.append(Input(f"pca_slider_{i}", "value"))

external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

def build_upload_layout():
    children = []
    children.append(
        dcc.Upload(
            id='upload-image-box',
            children=html.Div(id="drag_box", children=[
                'Drag and Drop or ',
                html.A('Select Files'),
                html.Div(id="output-image-upload")
            ]),
            multiple=False
        )
    )
    return html.Div(id="upload-layout", children=children)


def build_sliders():
    children = []
    for i in range(NUMBER_OF_PCA_SLIDERS):
        children.append(
            dcc.Slider(
                id=f"pca_slider_{i}",
                min=-1., max=1., step=0.01, value=0.0,
                tooltip={"always_visible": False, "placement": "bottom"}
            )
        )
    return children

def build_layout():
    layout = html.Div(className="container",
                      children=[
                          html.Div(className="menu-container",
                                   children=[
                                       html.H4(id="title", children=["Sustainable Deepfashion"]),
                                       html.P(id="description", children=[
                                           "This is a description, that is very long... In fact, it is so long that it spans over multile lines"]),
                                       build_upload_layout(),
                                       dcc.Graph(id="embedding-plot", config={"displayModeBar": False})
                                   ]),
                          html.Div(className="prediction-container",
                                   children=[
                                       html.Div(id="prediction-box", children=[
                                           html.Div(id='output-image-prediction'),
                                           html.Div(id='slider-container', children=build_sliders())
                                       ])
                                   ])
                      ])
    return layout





app.layout = html.Div(build_layout())


def parse_upload(contents):
    return [html.Center(html.Img(id="upload-image", src=contents))]


def predict_from_contents(contents, values):
    embedding = model.predict(contents)
    # embedding = None
    # print(convert_base64_string_to_array(contents))

    # values = np.array(values).flatten()
    # embedding = embedding + np.dot(values, pca_components)
    embedding = embedding.flatten()
    print(embedding, np.linalg.norm(embedding))
    prediction_df["distance"] = prediction_df["prediction"].apply(
        lambda x: distance(x, embedding, metric=model.get_metric())
    )

    if WEB_IMAGES:
        top_k_pred_images = prediction_df.sort_values(by="distance", ascending=True)["web_image"].head(
            NUMBER_OF_BEST_PREDICTIONS).to_list()
    else:
        top_k_pred = prediction_df.sort_values(by="distance", ascending=True)["image"].head(
            NUMBER_OF_BEST_PREDICTIONS).to_list()
        top_k_pred_images = ['data:image/jpeg;base64,{}'.format(base64.b64encode(open(file, 'rb').read()).decode()) for file in top_k_pred]

    return build_prediction_gallery(top_k_pred_images), update_embedding_plot(embedding)


def update_embedding_plot(embedding_vector):
    fig = go.Figure(go.Barpolar(
        r=[e + 1. for e in embedding_vector.tolist()],
        theta=[360. * i / embedding_vector.shape[0] for i in range(embedding_vector.shape[0])],
        marker_line_color="black",
        marker_line_width=1,
        opacity=1.0,
    ))

    fig.update_layout(
        # template=None,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=1, r=1, b=0, t=0),
        # width=300,
        height=250,
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks='', showgrid=False, showline=False),
            angularaxis=dict(showticklabels=False, ticks='', showgrid=False, showline=False)
        )
    )
    fig.update_polars(bgcolor="rgba(0,0,0,0)")
    return fig


def build_prediction_gallery(top_k_pred_base64):
    list_of_images = []
    for img in top_k_pred_base64:
        list_of_images.append(
            html.Li(
                html.Img(
                    id="prediction-img",
                    src=img
                )
            )
        )

    children = [html.Ul(children=list_of_images)]
    return children


@app.callback([Output('output-image-upload', 'children'),
               Output('output-image-prediction', 'children'),
               Output("embedding-plot", "figure")],
              Input('upload-image-box', 'contents'),
              slider_input)
def update_output(contents, *values):
    if contents is not None:
        return parse_upload(contents), *predict_from_contents(contents, values)
    else:
        return html.Img(id="default-image", src="assets/default.jpeg"), None, update_embedding_plot(np.zeros((EMBEDDING_SIZE)))


if __name__ == '__main__':
    app.run_server(debug=True)
