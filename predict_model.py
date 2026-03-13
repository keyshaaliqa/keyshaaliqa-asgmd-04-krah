import joblib
import pandas as pd


def predict(input_df):

    model = joblib.load("models/logreg_model.pkl")

    predictions = model.predict(input_df)

    return predictions