import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go

from dash.dependencies import Input

import numpy as np
import base64

from src.utility import load_components_from_basename
from src.inference import distance


class LayoutFactory:
    def __init__(self,
                 basename,
                 number_of_pca_sliders=3,
                 number_of_best_predictions=6,
                 web=False):
        self.model, self.prediction_df, self.pca_components = load_components_from_basename(basename)
        self.number_of_pca_sliders = number_of_pca_sliders
        self.number_of_best_predictions=number_of_best_predictions
        self.web = web

    def get_sliders_input(self):
        return [Input(f"pca_slider_{i}", "value") for i in range(self.number_of_pca_sliders)]

    def get_layout(self):
        layout = html.Div(className="container",
                          children=[
                              html.Div(className="menu-container",
                                       children=[
                                           html.H4(id="title", children=["Sustainable Deepfashion"]),
                                           html.P(id="description", children=[
                                               "This is a description, that is very long... In fact, it is so long that it spans over multile lines"]),
                                           self.build_upload_layout(),
                                           dcc.Graph(id="embedding-plot", config={"displayModeBar": False})
                                       ]),
                              html.Div(className="prediction-container",
                                       children=[
                                           html.Div(id="prediction-box", children=[
                                               html.Div(id='output-image-prediction'),
                                               html.Div(id='slider-container', children=self.build_sliders())
                                           ])
                                       ])
                          ])
        return layout

    def build_upload_layout(self):
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

    def build_sliders(self):
        children = []
        for i in range(self.number_of_pca_sliders):
            children.append(
                dcc.Slider(
                    id=f"pca_slider_{i}",
                    min=-1., max=1., step=0.01, value=0.0,
                    tooltip={"always_visible": False, "placement": "bottom"}
                )
            )
        return children


    def build_prediction_gallery(self, top_k_pred_base64):
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

    @staticmethod
    def build_uploaded_image(contents):
        return [html.Center(html.Img(id="upload-image", src=contents))]

    @staticmethod
    def update_embedding_plot(embedding_vector):
        fig = go.Figure(go.Barpolar(
            r=[e + 1. for e in embedding_vector.tolist()],
            theta=[360. * i / embedding_vector.shape[0] for i in range(embedding_vector.shape[0])],
            marker_line_color="black",
            marker_line_width=1,
            opacity=1.0,
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=1, r=1, b=0, t=0),
            height=250,
            polar=dict(
                radialaxis=dict(showticklabels=False, ticks='', showgrid=False, showline=False),
                angularaxis=dict(showticklabels=False, ticks='', showgrid=False, showline=False)
            )
        )
        fig.update_polars(bgcolor="rgba(0,0,0,0)")
        return fig

    def predict_from_contents(self, contents, values):
        embedding = self.model.predict(contents)

        values = np.array(values).flatten()
        embedding = embedding + np.dot(values, self.pca_components[:self.number_of_pca_sliders])
        embedding = embedding.flatten()

        self.prediction_df["distance"] = self.prediction_df["prediction"].apply(
            lambda x: distance(x, embedding, metric=self.model.get_metric())
        )

        if self.web:
            top_k_pred_images = self.prediction_df.sort_values(by="distance", ascending=True)["web_image"].head(
                self.number_of_best_predictions).to_list()
        else:
            top_k_pred = self.prediction_df.sort_values(by="distance", ascending=True)["image"].head(
                self.number_of_best_predictions).to_list()
            top_k_pred_images = ['data:image/jpeg;base64,{}'.format(base64.b64encode(open(file, 'rb').read()).decode()) for
                                 file in top_k_pred]

        return self.build_prediction_gallery(top_k_pred_images), self.update_embedding_plot(embedding)

