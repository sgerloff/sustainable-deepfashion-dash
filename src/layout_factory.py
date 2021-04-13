import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
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
        layout = html.Div(className="container-fluid", id="layout", children=[
            self.build_title_logo(),
            dbc.Row([
                dbc.Col(
                    dbc.Collapse(
                        self.build_info_page(),
                        id="learn-more-collapse"
                    ),
                    xs=12, md=12, lg=9
                )
            ]),
            dbc.Row([
                self.build_description(),
                self.build_upload_layout(),
                self.build_embedding_plot_container()
            ], justify="center"),
            dbc.Row([
                dbc.Col(
                    id="prediction-block"
                )
            ]),
            dbc.Row(
                dbc.Col(dbc.Card(id="slider-container", children=self.build_sliders()), xs=12, md=12, lg=9),
                justify="center"
            ),
            html.Div(id="intermediate-embedding")
        ])
        return layout

    @staticmethod
    def build_title_logo():
        return dbc.Row(
            id="title-row",
            children=dbc.Col(
                html.Center(
                    html.A(
                        html.Img(id="title-logo", src="assets/sustainable_deepfashion_title.png"),
                        href="https://github.com/sgerloff/sustainable_deepfashion"
                    )
                ),
                xs=6,
                md=4,
                lg=4,
                xl=3
            ),
            justify="center"
        )

    @staticmethod
    def build_info_page():
        return dbc.Card(
            dbc.CardBody([
                html.H5("Concept - What's Happening?"),
                html.P(
                    "Using deep-learning, we have trained a model to understand similarity between \
                    short-sleeved tops. \
                    Comparing an user-provided image to a database comprised of second-hand items, we \
                    return the most similar sustainable alternatives. \
                    Here, similarity refers to the prints, patterns and colors of the item, instead of \
                    the angle, lighting and crop of the corresponding image. \
                    Furthermore, we allow users to modify the resulting representations introduce new \
                    features, like floral patterns, stripes and dots.",
                    style={"text-align": "justify"}
                ),
                html.H5("Data - Where does it come from?"),
                html.P(
                    "All predictions are pulled from a small demo database, containing a random picture \
                    of over 10000 different items from our training- and testing dataset. \
                    Currently, it focuses on short-sleeved tops, but the approach can be easily generalized \
                    to different categories.",
                    style={"text-align": "justify"}
                ),
                html.P(
                    "We use data from the Deepfashion2 dataset, pictures taken by ourselves, friends and family, \
                    as well as pictures scraped from Vinted (a.k.a. Kleiderkreisel), a popular german platform \
                    for second-hand fashion. \
                    In total, our dataset contains up to 500.000 pictures of more then 10.000 items, with different \
                    angles, framing, lighting and so on. \
                    We have chosen this data to reflect the distribution of images that you could find on the second-\
                    hand market.",
                    style={"text-align": "justify"}
                ),
                html.H5("Model - How does it work?"),
                html.P(
                    "Our model is built with a custom architecture, containing five convolutional layers followed \
                    by three dense layers. From input images the model predicts representation vectors embedded \
                    into an 20-dimensional latent space. These representations have the property that their angular \
                    distances are small if the images contain the same item and large if they differ. \
                    The similarity between two items is then defined accordingly as the distance between the \
                    representations vectors in latent space.",
                    style={"text-align": "justify"}
                ),
                html.P(
                    "This is achieved by training the model on the semi-hard triplet loss, which we have \
                    implemented in Tensorflow and trained on GPU's provided by AWS and Google Cloud. \
                    To find the best performing model, we compare the Top-K Accuracy of the model to retrieve \
                    the matching item from a validation and test dataset. \
                    To this end, we have explored pretrained models as feature extractors, different distance \
                    metrics, other dimensionality of the latent space, and more.",
                    style={"text-align": "justify"}
                ),
                html.H5("More - Who are we?"),
                html.P(
                    "We are three Data Scientists, that have met at Data Science Retreat and set out to improve \
                    the second-hand fashion market. If you have questions, feedback or want to chat, please feel \
                    free to contact us:",
                    style={"text-align": "justify"}
                ),
                html.Ul([
                    html.Li(html.A("Gert-Jan Dobbelaere", href="https://www.linkedin.com/in/gert-jan-dobbelaere/")),
                    html.Li(html.A("Dr. Sascha Gerloff", href="https://www.linkedin.com/in/sascha-gerloff/")),
                    html.Li(html.A("Sergio Vechi", href="https://www.linkedin.com/in/sergiovechi/"))
                ]),
                html.P(
                    "If you want to learn more about the implementation details, or want to incorporate \
                    some of our results into your own project, you can find all the source code at github:",
                    style={"text-align": "justify"}
                ),
                html.Ul(html.Li(html.A(
                    "https://github.com/sgerloff/sustainable_deepfashion",
                    href="https://github.com/sgerloff/sustainable_deepfashion"
                ))),
                html.P("Interested how the dash app is build? We got you covered:", style={"text-align": "justify"}),
                html.Ul(html.Li(html.A(
                    "https://github.com/sgerloff/sustainable-deepfashion-dash",
                    href="https://github.com/sgerloff/sustainable-deepfashion-dash"
                )))
            ], id="info-page"),
            style={"margin-bottom": "15px"}
        )

    @staticmethod
    def build_description():
        return dbc.Col(
            html.Div(
                id="description-text",
                children=[
                    html.P("""
                    Second-hand fashion solves many ethical- and environmental problems posed by the fast fashion industry. However, finding items that suit your style can be hard!
                    """, style={"text-align": "justify"}),
                    html.P("""
                    This AI-powered app demonstrates how we can help! Simply upload an image of the desired style and get the best alternatives from second-hand sources.
                    """, style={"text-align": "justify"}),
                    dbc.Button(
                        "Learn More",
                        id="learn-more-button",
                        className="mb-3"
                    )
                ]),
            xs=12,
            md=4,
            lg=3
        )

    def build_upload_layout(self):
        return dbc.Col(
            children=[
                html.Center(html.H5("Upload")),
                dcc.Loading(
                    id="loading-upload",
                    type="default",
                    children=self.build_upload_image()
                )
            ],
            xs=6, md=4, lg=3
        )

    @staticmethod
    def build_upload_image():
        return dcc.Upload(
            id="upload-container",
            multiple=False,
            children=[
                html.Div(className="square-container", id="square-upload-container")
            ]
        )

    @staticmethod
    def image_from_content(contents):
        return html.Img(id="upload-image", src=contents)

    @staticmethod
    def upload_button():
        return html.Img(id="upload-button", src="assets/plus-symbol.png")

    @staticmethod
    def build_embedding_plot_container():
        return dbc.Col([
            html.Center(html.H5(id="representation-description")),
            html.Div(id="embedding-plot-container", className="square-container")
        ], xs=6, md=4, lg=3)

    def build_sliders(self):
        children = [
            html.H5(className="description", children=[
                "Modify and Explore:"
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
                ), className="single-slider-container")
            )

        for k in self.pattern_sliders.keys():
            children.append(
                html.Center(dcc.Slider(
                    id=f"{k}_slider",
                    className="pattern_slider",
                    min=0, max=2., step=0.1, value=0.0,
                    tooltip={"always_visible": False, "placement": "bottom"},
                    marks={0: "original",
                           2: k}
                ), className="single-slider-container")
            )

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

        children = [
            # html.Center([
            #     html.H5("\U0001F814 Top-10 Second Hand Items \U0001F816"),
            # ]),
            html.Div(id="prediction-container", children=list_of_images)
        ]

        return children

    @staticmethod
    def build_embedding_plot(embedding_vector):
        fig = go.Figure(go.Barpolar(
            r=[e + 1. for e in embedding_vector.tolist()],
            theta=[360. * i / embedding_vector.shape[0] for i in range(embedding_vector.shape[0])],
            marker_line_color="#051A29",
            marker_line_width=1,
            marker_color="#05CEB9",
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
