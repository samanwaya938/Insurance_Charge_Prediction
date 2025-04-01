import os
import sys
from src.exception import MyException
from src.logger import logging
from src.config import ModelTrainerConfig
from src.components.model_evaluation import ModelEvaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from src.utils import save_object, read_yaml_file

class ModelTraining:
  def __init__(self):
    self.model_training_config = ModelTrainerConfig()
    self.model_evaluation = ModelEvaluation()

  def initiate_model_training(self, train_arr, test_arr):
    try:
      X_train, y_train, X_test, y_test = train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1]
      logging.info("Model Training started")

      models = {
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR()
      }

      params = read_yaml_file(file_path="params.yaml")
      print(f"Loaded params from YAML: {params}")  # Debug statement
      if not params:
            raise Exception("Params.yaml is empty or not loaded correctly")

      report, mae, mse = self.model_evaluation.evaluate_models(X_train,y_train,X_test,y_test,models,params)
      print(f"Model Report : {report}")

      logging.info("Model training completed")

      best_model_score = max(sorted(report.values()))
      print(f"Best Model Score : {best_model_score}")

      best_model_name = max(report, key=lambda k: report[k])
      print(f"Best Model Name : {best_model_name}")
      
      best_model = models[best_model_name]
      
      if best_model_score < self.model_training_config.base_accuracy:
        raise Exception("No best model found")

      
      save_object(
        file_path=self.model_training_config.trained_model_file_path,
        obj=best_model
      )

      logging.info("Model saved")

      return (
        self.model_training_config.trained_model_file_path
      )
    except Exception as e:
      raise MyException(e, sys)