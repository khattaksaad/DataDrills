import pytest
import pandas as pd
from main import model, species_mapping


def test_model_prediction():
    """
    Test the model prediction.
    """
    input_data = pd.DataFrame({
        "sepal_length": [5.1],
        "sepal_width": [3.5],
        "petal_length": [1.4],
        "petal_width": [0.2]
    })
    prediction = model.predict(input_data)[0]
    species = species_mapping.get(prediction, "Unknown")
    assert species == "setosa", "The prediction did not match the expected output"

