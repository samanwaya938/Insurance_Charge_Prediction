import sys
from src.exception import MyException
from src.components.data_ingestion import DataIngestion

if __name__ == '__main__':
  try:
    data_ingestion = DataIngestion()
    train_file_path, test_file_path = data_ingestion.initiate_data_ingestion()
  except Exception as e:
    raise MyException(e, sys)