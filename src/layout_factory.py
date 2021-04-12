import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go

from dash.dependencies import Input

import numpy as np
import base64, joblib

from src.utility import load_components_from_basename
from src.inference import distance


class LayoutFactory:
    def __init__(self,
                 basename,
                 number_of_pca_sliders=1,
                 number_of_best_predictions=6,
                 web=False):
        self.content_embedding = None;
        self.slider_values = None;

        self.pattern_sliders = joblib.load("slider_direction_dict.joblib")
        self.pattern_directions = [np.expand_dims(v, axis=0) for _, v in self.pattern_sliders.items()]

        self.model, self.prediction_df, self.pca_components = load_components_from_basename(basename)
        self.number_of_pca_sliders = number_of_pca_sliders
        self.number_of_best_predictions = number_of_best_predictions
        self.web = web

    def get_sliders_input(self):
        sliders = [Input(f"pca_slider_{i}", "value") for i in range(self.number_of_pca_sliders)]
        sliders.extend([Input(f"{k}_slider", "value") for k in self.pattern_sliders.keys()])
        return sliders

    def get_layout(self):
        layout = html.Div(className="app_container",
                          children=[
                              self.build_menu_container(),
                              self.build_prediction_container(),
                              html.Div(id='slider-container', children=self.build_sliders()),
                              html.Div(id="intermediate-embedding", style={"display": "none"})
                          ])
        return layout

    def build_menu_container(self):
        return html.Div(className="menu-container",
                        children=[
                            self.build_title(),
                            self.build_description(),
                            html.Center(
                                className="upload-image-container",
                                children=[
                                    self.build_upload_layout(),
                                    html.Div(id="embedding-plot-container")
                                ])
                            # html.Div(id="embedding-plot-container"),
                        ])

    def build_title(self):
        return html.H4(id="title", children=["Sustainable Deepfashion"])

    def build_description(self):
        return html.P(className="description", children=[
            """
            Upload an image of a short-sleeved top to get alternatives from second-hand sources.
            Simply press the button or drag and drop:
            """
        ])

    def build_prediction_container(self):
        return html.Div(id="prediction-box", children=[
            html.P(className="description", id="demo-description"),
            html.Div(id='output-image-prediction'),
            # html.Div(id='slider-container', children=self.build_sliders())
        ])

    def build_demo_description(self):
        return [f"Top {self.number_of_best_predictions} second-hand alternatives:"]

    def build_upload_layout(self):
        children = [
            dcc.Upload(
                id='upload-image-box',
                multiple=False
            )
        ]

        return html.Div(id="upload-layout", children=[
            dcc.Loading(id="loading-upload",
                        children=children,
                        type="default")
        ])

    def build_default_image(self):
        return html.Img(className="upload-image", src="assets/default.jpeg")

    def build_sliders(self):
        children = [
            html.P(className="description", children=[
                "Modify your predictions:"
            ])
        ]

        mark_dict = {
            0: {
                -1: "brighter",
                1: "darker"
            },
            1: {
                -1: "unknown",
                1: "unknown"
            },
            2: {
                -1: "unknown",
                1: "unknown"
            }
        }

        for i in range(self.number_of_pca_sliders):
            children.append(
                html.Center(dcc.Slider(
                    id=f"pca_slider_{i}",
                    min=-1., max=1., step=0.1, value=0.0,
                    tooltip={"always_visible": False, "placement": "bottom"},
                    marks=mark_dict[i]
                ))
            )

        for k in self.pattern_sliders.keys():
            children.append(
                html.Center(dcc.Slider(
                    id=f"{k}_slider",
                    min=0, max=2., step=0.1, value=0.0,
                    tooltip={"always_visible": False, "placement": "bottom"},
                    marks={0: "original",
                           2: k}
                ))
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

    def build_simple_gallery(self, top_k_pred_base64):
        list_of_images = []
        for img in top_k_pred_base64:
            list_of_images.append(
                html.Img(
                    id="prediction-img",
                    src=img
                )
            )

        return list_of_images

    @staticmethod
    def build_uploaded_image(contents):
        return html.Img(className="upload-image", src=contents)

    @staticmethod
    def build_embedding_plot(embedding_vector):
        fig = go.Figure(go.Barpolar(
            r=[e + 1. for e in embedding_vector.tolist()],
            theta=[360. * i / embedding_vector.shape[0] for i in range(embedding_vector.shape[0])],
            marker_line_color="#393C3D",
            marker_line_width=1,
            marker_color="#1DA1F2",
            opacity=0.8
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=0),
            polar=dict(
                radialaxis=dict(showticklabels=False, ticks='', showgrid=False, showline=False),
                angularaxis=dict(showticklabels=False, ticks='', showgrid=False, showline=False)
            )
        )
        fig.update_polars(bgcolor="rgba(0,0,0,0)")
        return dcc.Graph(id="embedding-plot", figure=fig, config={"displayModeBar": False})

    def predict_from_contents(self, embedding, values):
        embedding = np.array(embedding)

        embedding = self.add_slider_values(embedding, values)
        self.compute_distances(embedding)

        if self.web:
            top_k_pred_images = self.get_top_k_img_from_web()
        else:
            top_k_pred_images = self.get_top_k_img_from_disk()

        return self.build_simple_gallery(top_k_pred_images), self.build_embedding_plot(embedding)

    def infer_prediction_from_contents(self, contents):
        return self.model.predict(contents).flatten().tolist()

    def add_slider_values(self, embedding, values):
        values = np.array(values).flatten()
        directions = self.get_directions()

        embedding_delta = np.dot(values, directions)
        embedding = embedding + embedding_delta
        embedding = embedding.flatten()
        return embedding / np.linalg.norm(embedding)

    def get_directions(self):
        directions = [self.pca_components[:self.number_of_pca_sliders]]
        directions.extend(self.pattern_directions)
        return np.concatenate(directions)

    def compute_distances(self, embedding):
        self.prediction_df["distance"] = self.prediction_df["prediction"].apply(
            lambda x: distance(x, embedding, metric=self.model.get_metric())
        )

    def get_top_k_img_from_web(self):
        return self.prediction_df.sort_values(by="distance", ascending=True)["web_image"].head(
            self.number_of_best_predictions).to_list()

    def get_top_k_img_from_disk(self):
        top_k_pred = self.prediction_df.sort_values(by="distance", ascending=True)["image"].head(
            self.number_of_best_predictions).to_list()
        return ['data:image/jpeg;base64,{}'.format(base64.b64encode(open(file, 'rb').read()).decode()) for file in
                top_k_pred]
