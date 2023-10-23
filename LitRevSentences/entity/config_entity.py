from dataclasses import dataclass
from LitRevSentences.constants import *
import os

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.DATASET_NAME: str = DATASET_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR)
        self.DATASET_ARTIFACTS_DIR: str =  os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATASET_NAME)
        #self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        #self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,self.ZIP_FILE_NAME)
        self.DATASET_GITHUB_LINK = os.path.join(DATASET_GitHub,DATASET_NAME)
        self.DATASET_PATH = os.path.join(os.getcwd(),DATASET_DIR)
        self.TRAIN_DATA_PATH = os.path.join(self.DATASET_PATH,TRAIN_FILE)
        self.TEST_DATA_PATH = os.path.join(self.DATASET_PATH,TEST_FILE)
        self.DEV_DATA_PATH = os.path.join(self.DATASET_PATH,DEV_FILE)


"""
@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE 
        self.DROP_COLUMNS = DROP_COLUMNS
        self.CLASS = CLASS 
        self.LABEL = LABEL
        self.TWEET = TWEET


@dataclass
class ModelTrainerConfig: 
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR) 
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR,TRAINED_MODEL_NAME)
        self.X_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TEST_FILE_NAME)
        self.Y_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, Y_TEST_FILE_NAME)
        self.X_TRAIN_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TRAIN_FILE_NAME)
        self.MAX_WORDS = MAX_WORDS
        self.MAX_LEN = MAX_LEN
        self.LOSS = LOSS
        self.METRICS = METRICS
        self.ACTIVATION = ACTIVATION
        self.LABEL = LABEL
        self.TWEET = TWEET
        self.RANDOM_STATE = RANDOM_STATE
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.VALIDATION_SPLIT = VALIDATION_SPLIT



@dataclass
class ModelEvaluationConfig: 
    def __init__(self):
        self.MODEL_EVALUATION_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR_PATH: str = os.path.join(self.MODEL_EVALUATION_MODEL_DIR,BEST_MODEL_DIR)
        self.BEST_MODEL_DIR_PATH_SOURCE: str = os.path.join(os.getcwd(),BEST_MODEL_DIR,MODEL_NAME)
        self.MODEL_NAME = MODEL_NAME 
        self.src_file_path = os.path.join(os.getcwd(),ZIP_FILE_NAME)
        self.BEST_MODEL_PATH = os.path.join(self.MODEL_EVALUATION_MODEL_DIR,BEST_MODEL_DIR) #TODO Make sure correct. Change back to os.getcwd() if error



@dataclass
class ModelPusherConfig:

    def __init__(self):
        self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, MODEL_NAME)
        self.MODEL_NAME = MODEL_NAME
        self.BEST_MODEL_PATH = os.path.join(os.getcwd(),BEST_MODEL_DIR)
        
    """