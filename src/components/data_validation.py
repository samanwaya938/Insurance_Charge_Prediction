import os
import sys
import yaml
import pandas as pd
from src.exception import MyException
from src.logger import logging
from src.config import DataValidationConfig
from src.utils import read_yaml_file, write_yaml_file
class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()
        
    def validate_schema(self, file_path) -> bool:
        try:
            validation_status = True
            
            # Load schema
            # with open(self.validation_config.schema_file_path, 'r') as f:
            #     schema = yaml.safe_load(f)

            schema = read_yaml_file(self.validation_config.schema_file_path)
            
            # Read data
            df = pd.read_csv(file_path)
            
            # Check column names and count
            if not df.columns.equals(pd.Index(schema['columns'].keys())):
                logging.error("Column name mismatch found")
                validation_status = False
                
            # Check data types
            for column, dtype in schema['columns'].items():
                if df[column].dtype != dtype:
                    logging.error(f"Datatype mismatch for {column}: Expected {df[column].dtype}, Found {dtype}")
                    validation_status = False
                    
            return validation_status
            
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self,train_file_path,test_file_path):
        try:
            logging.info("Starting Data Validation")
            
            os.makedirs(os.path.dirname(self.validation_config.report_file_path), exist_ok=True)
            
            validation_results = {
                "train_data": self.validate_schema(train_file_path),
                "test_data": self.validate_schema(test_file_path)
            }
            
            # Write validation report
            # with open(self.validation_config.report_file_path, 'w') as f:
            #     yaml.dump(validation_results, f)

            write_yaml_file(self.validation_config.report_file_path, validation_results)
                
            logging.info(f"Validation report saved to {self.validation_config.report_file_path}")
            
            return validation_results
            
        except Exception as e:
            raise MyException(e, sys)