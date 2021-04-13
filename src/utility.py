from dash.dependencies import Input
import dash_html_components as html

from src.inference import ModelInference
import joblib, os
import numpy as np
import pandas as pd


def load_components_from_basename(basename):
    model = ModelInference(os.path.join("models", basename, f"{basename}_saved_model"))

    prediction_csv_path = os.path.join("models", basename, f"{basename}_predictions.csv")
    prediction_df = pd.read_csv(prediction_csv_path)
    prediction_df["prediction"] = prediction_df["prediction"].apply(lambda x: np.array(eval(x)).flatten())

    pca_components = joblib.load(os.path.join("models", basename, f"{basename}_pca.joblib"))

    return model, prediction_df, pca_components


def get_sliders_input(number_of_sliders):
    return [Input(f"pca_slider_{i}", "value") for i in range(number_of_sliders)]


