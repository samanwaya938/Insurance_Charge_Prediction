import os
import sys
import mlflow
import dagshub
import numpy as np
import tempfile
from src.exception import MyException
from src.logger import logging
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

class ModelEvaluation:
  def __init__(self):        
    dagshub.init(repo_owner='samanwaya938', repo_name='Insurance_Charge_Prediction', mlflow=True)

    mlflow.set_tracking_uri("https://dagshub.com/samanwaya938/Insurance_Charge_Prediction.mlflow") # Add if using a remote server
    mlflow.set_experiment("Dagshub_Insurance_Experiment")
    logging.info("Initialized Dagshub Experiment tracking")

  def evaluate_models(self,X_train,y_train,X_test,y_test,models,params):

    try:
      report = {}

      for model_name, model in models.items():  # Iterate directly over model names and objects
          param = params.get(model_name, {})  # Get parameters for the current mode
          print(f"Parameters for {model_name}: {param}")

          if not param:              
              print(f"⚠️ No parameters found for {model_name} in params.yaml")


          rf = RandomizedSearchCV(model, param, cv=3, n_jobs=-1, verbose=2)
          rf.fit(X_train, y_train)
          print(f"\n{model_name} Best Parameters:")
          print(rf.best_params_)

          # Update model with best parameters
          best_model = model.set_params(**rf.best_params_)
          best_model.fit(X_train, y_train)
          # Predictions
          y_test_pred = best_model.predict(X_test)
          # R2 Score
          test_model_score = r2_score(y_test, y_test_pred)
          # Mean Absolute Error
          mae = mean_absolute_error(y_test, y_test_pred)
          # Mean Squared Error
          mse = mean_squared_error(y_test, y_test_pred)
          # Store score in the report dictionary
          report[model_name] = test_model_score

          with mlflow.start_run(run_name=model_name):                    
                    mlflow.log_params(rf.best_params_)                    
                    mlflow.log_metric("r2_score", test_model_score)

                    # Log mse and mae as artifacts
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        mae_path = os.path.join(tmp_dir, "Mean_absolute_error.txt")
                        np.savetxt(mae_path, [mae], fmt='%.5f')
                        mlflow.log_artifact(mae_path, "Mean_absolute_error")

                        mse_path = os.path.join(tmp_dir, "mean_squared_error.txt")
                        np.savetxt(mse_path, [mse], fmt='%.5f')
                        mlflow.log_artifact(mse_path, "Mean_squared_error")

                    # Log the trained model
                    mlflow.sklearn.log_model(sk_model=best_model,
                        artifact_path=model_name,
                        registered_model_name=model_name)
      return report, mae, mse


    except Exception as e:
      raise MyException(e, sys)