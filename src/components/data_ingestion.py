import os
import pandas as pd
import sys
from src.exception import MyException
from src.logger import logging
from src.config import DataIngestionConfig
from sklearn.model_selection import train_test_split

class DataIngestion:
  def __init__(self):
    self.data_ingestion_config = DataIngestionConfig()

  def initiate_data_ingestion(self):
    try:
      df = pd.read_csv(r"data\insurance.csv")
      logging.info("Read the dataset as dataframe")

      os.makedirs(os.path.dirname(self.data_ingestion_config.train_file_path), exist_ok=True)

      df.to_csv(self.data_ingestion_config.raw_file_path, index=False, header=True)

      logging.info("Train test split initiated")

      train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

      train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
      test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

      logging.info("Ingestion of the data is completed")
  
      return (
        self.data_ingestion_config.train_file_path,
        self.data_ingestion_config.test_file_path
      )      
      
    except Exception as e:
      raise MyException(e, sys)
