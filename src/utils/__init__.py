import os
import yaml
import sys
import pickle
import yaml
from src.exception import MyException

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
      pickle.dump(obj, file_obj) 
  except Exception as e:
    raise MyException(e, sys)
  
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise MyException(e,sys)
    
def read_yaml_file(file_path: str) -> dict:
  try:
      with open(file_path, "rb") as yaml_file:
          return yaml.safe_load(yaml_file)

  except Exception as e:
      raise MyException(e, sys) from e
  
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MyException(e, sys) from e