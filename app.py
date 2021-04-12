import dash
from dash.dependencies import Input, Output
import dash_html_components as html

import numpy as np
from src.layout_factory import LayoutFactory

import json

external_stylesheets = [
    "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css",
    {
        "rel": "preconnect",
        "href": "https://fonts.gstatic.com"
    },
    {
        "href": "https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap",
        "rel": "stylesheet"
    }
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

basename = "simple_conv2d_embedding_size_20_angular_d_augmented"
# basename = "VAE_conv2d_input_224_embedding_512"
layout_factory = LayoutFactory(basename,
                               number_of_pca_sliders=1,
                               number_of_best_predictions=10,
                               web=True)
app.layout = layout_factory.get_layout()


@app.callback([
    Output("upload-image-box", "children"),
    Output("intermediate-embedding", "children")
],
    Input("upload-image-box", "contents")
)
def update_upload(contents):
    if contents is not None:
        intermediate_embedding = layout_factory.infer_prediction_from_contents(contents)
        return layout_factory.build_uploaded_image(contents), json.dumps(intermediate_embedding)
    else:
        return layout_factory.build_default_image(), None


@app.callback(
    [
        Output('output-image-prediction', 'children'),
        Output("embedding-plot-container", "children"),
        Output("slider-container", "style"),
        Output("demo-description", "children")
    ],
    Input('intermediate-embedding', 'children'),
    layout_factory.get_sliders_input()
)
def update_predictions(children, *values):
    if children is not None:
        embedding = json.loads(children)
        output_gallery, embedding_plot = layout_factory.predict_from_contents(embedding, values)
        return output_gallery, embedding_plot, {'display': 'block'}, layout_factory.build_demo_description()
    else:
        return None, None, {'display': 'none'}, None


if __name__ == '__main__':
    app.run_server(debug=True)
