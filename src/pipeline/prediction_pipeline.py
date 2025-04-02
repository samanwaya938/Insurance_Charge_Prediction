import sys
import os
import pandas as pd
from src.exception import MyException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Define paths to the preprocessor and model files
            preprocessor_path = os.path.join("artifact", "data_transformation", "preprocessing.pkl")
            model_path = os.path.join("artifact", "model_trainer", "model.pkl")

            # Load the preprocessor and model objects
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Make a prediction using the model
            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise MyException(e, sys)

class CustomData:
    def __init__(self,
                 age: int,
                 sex: str,
                 bmi: float,
                 children: int,
                 smoker: str,
                 region: str):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_dataframe(self):
        try:
            # Create a dictionary with insurance-related features
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region]
            }

            # Return the data as a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise MyException(e, sys)