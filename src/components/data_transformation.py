import os
import sys
import numpy as np
from src.logger import logging
from src.exception import MyException
import pandas as pd
from src.config import DataTransformationConfig
from src.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class DataTransformation:
  def __init__(self):
    self.data_trasnsformation_config = DataTransformationConfig()

  def get_data_transformation(self):
    try:
      numeric_colsumns = ['age', 'bmi', 'children']
      categorical_columns = ['sex', 'smoker', 'region']

      numeric_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
      ])

      categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder())
      ])

      preprocessor = ColumnTransformer([
        ('num_pipeline', numeric_pipeline, numeric_colsumns),
        ('cat_pipeline', categorical_pipeline, categorical_columns)
      ])

      return preprocessor

  
    except Exception as e:
      raise MyException(e, sys)

  def initiate_data_transformation(self, train_file_path, test_file_path):
    try:
      train_df = pd.read_csv(train_file_path)
      test_df = pd.read_csv(test_file_path)
      logging.info("Read train and test data completed")

      preprocessing_obj = self.get_data_transformation()

      input_feature_train_df = train_df.drop(columns=['charges','index'], axis=1)
      target_feature_train_df = train_df['charges']

      input_feature_test_df = test_df.drop(columns=['charges','index'], axis=1)
      target_feature_test_df = test_df['charges']

      logging.info("Splitting training and test input and target feature")

      # After the Columntransformer the combined feature desnsity is higher than the sparse_threshold (default is 0.3), so the output is Dense Matrix instead of Sparse so we do not need to use .toarray() function to convert it to dense matrix

      input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

      train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
      test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

      save_object(
        file_path=self.data_trasnsformation_config.preproccesing_file_path,
        obj=preprocessing_obj
      )

      return (
        train_arr,
        test_arr,
        self.data_trasnsformation_config.preproccesing_file_path
      )

    except Exception as e:
      raise MyException(e, sys)

  


