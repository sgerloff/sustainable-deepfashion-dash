import dash, json
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from src.layout_factory import LayoutFactory

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    {
        "rel": "preconnect",
        "href": "https://fonts.gstatic.com"
    },
    {
        "href": "https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap",
        "rel": "stylesheet"
    }
]

basename = "simple_conv2d_embedding_size_20_angular_d_augmented"

layout_factory = LayoutFactory(basename,
                               number_of_pca_sliders=1,
                               number_of_best_predictions=10,
                               web=False)

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                meta_tags=[
                    {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}
                ])
server = app.server

app.layout = layout_factory.get_layout()


@app.callback([
    Output("square-upload-container", "children"),
    Output("intermediate-embedding", "children"),
    Output("representation-description", "children")
],
    Input("upload-container", "contents")
)
def update_upload(contents):
    if contents is not None:
        intermediate_embedding = layout_factory.infer_prediction_from_contents(contents)
        return layout_factory.image_from_content(contents), json.dumps(intermediate_embedding), "Representation"
    else:
        return layout_factory.upload_button(), None, None


@app.callback(
    [
        Output('prediction-block', 'children'),
        Output("embedding-plot-container", "children"),
        Output("slider-container", "style")
    ],
    Input('intermediate-embedding', 'children'),
    layout_factory.get_sliders_input()
)
def update_predictions(children, *values):
    if children is not None:
        embedding = json.loads(children)
        output_gallery, embedding_plot = layout_factory.predict_from_contents(embedding, values)
        return output_gallery, embedding_plot, {'display': 'block'}
    else:
        return None, None, {'display': 'none'}


@app.callback(
    Output("learn-more-collapse", "is_open"),
    [Input("learn-more-button", "n_clicks")],
    [State("learn-more-collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=True)
