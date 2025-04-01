import sys
from src.exception import MyException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

if __name__ == '__main__':
  try:
    data_ingestion = DataIngestion()
    train_file_path, test_file_path = data_ingestion.initiate_data_ingestion()

    data_validation = DataValidation()
    data_validation.initiate_data_validation(train_file_path, test_file_path)

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_file_path, test_file_path)

    model_training = ModelTraining()
    model_training.initiate_model_training(train_arr, test_arr)
  except Exception as e:
    raise MyException(e, sys)