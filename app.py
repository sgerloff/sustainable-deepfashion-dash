import dash
from dash.dependencies import Input, Output
import dash_html_components as html

import numpy as np
from src.layout_factory import LayoutFactory



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

basename = "simple_conv2d_embedding_size_16_angular_d-0"
# basename = "VAE_conv2d_input_224_embedding_512"
layout_factory = LayoutFactory(basename,
                               number_of_pca_sliders=3,
                               number_of_best_predictions=10,
                               web=False)
app.layout = layout_factory.get_layout()


@app.callback([Output('upload-image-box', 'children'),
               Output('output-image-prediction', 'children'),
               Output("embedding-plot-container", "children"),
               Output("slider-container", "style"),
               Output("demo-description", "children")],
              Input('upload-image-box', 'contents'),
              layout_factory.get_sliders_input())
def update_output(contents, *values):
    if contents is not None:
        output_gallery, embedding_plot = layout_factory.predict_from_contents(contents, values)
        return layout_factory.build_uploaded_image(contents), output_gallery, embedding_plot, {'display': 'block'}, layout_factory.build_demo_description()
    else:
        # return html.Img(id="default-image", src="assets/default.jpeg"), None, layout_factory.update_embedding_plot(
            # np.zeros((layout_factory.pca_components.shape[1])))
        return layout_factory.build_default_image(), None, None, {'display': 'none'}, None



if __name__ == '__main__':
    app.run_server(debug=True)
