import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
  raw_file_path = os.path.join("artifact","data_ingestion","raw_data.csv")
  train_file_path = os.path.join("artifact","data_ingestion","train.csv")
  test_file_path = os.path.join("artifact","data_ingestion","test.csv")


@dataclass(frozen=True)
class DataValidationConfig:
    schema_file_path: str = os.path.join("config","schema.yaml")
    report_file_path: str = os.path.join("artifact","data_validation","validation_report.txt")