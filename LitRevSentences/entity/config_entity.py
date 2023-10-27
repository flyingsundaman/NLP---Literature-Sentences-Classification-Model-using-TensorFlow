from dataclasses import dataclass
from LitRevSentences.constants import *
import os

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.DATASET_NAME: str = DATASET_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR)
        self.TRAIN_DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR , TRAIN_FILE)
        self.TEST_DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR , TEST_FILE)
        self.VAL_DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR , DEV_FILE)
        #self.DATASET_ARTIFACTS_DIR: str =  os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATASET_NAME)
        #self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        #self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,self.ZIP_FILE_NAME)
        self.DATASET_GITHUB_LINK = os.path.join(DATASET_GitHub,DATASET_NAME)
        self.DATASET_PATH = os.path.join(os.getcwd(),DATASET_DIR)
        self.TRAIN_DATA_PATH = os.path.join(self.DATASET_PATH,TRAIN_FILE)
        self.TEST_DATA_PATH = os.path.join(self.DATASET_PATH,TEST_FILE)
        self.VAL_DATA_PATH = os.path.join(self.DATASET_PATH,DEV_FILE)



@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_SENTENCES_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRAIN_SENTENCES_NAME)
        self.TEST_SENTENCES_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TEST_SENTENCES_NAME)
        self.VAL_SENTENCES_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,VAL_SENTENCES_NAME)
        self.TRAIN_ONE_HOT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRAIN_ONE_HOT_NAME)
        self.TEST_ONE_HOT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TEST_ONE_HOT_NAME)
        self.VAL_ONE_HOT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,VAL_ONE_HOT_NAME)
        self.TRAIN_DATASET = None
        self.TEST_DATASET = None
        self.VAL_DATASET = None

@dataclass
class ModelTrainerConfig: 
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR) 
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR, MODEL_NAME)
        self.MAX_TOKENS = MAX_TOKENS
        self.OUTPUT_SEQUENCE_LENGTH = OUTPUT_SEQUENCE_LENGTH
        self.length_rct_20k_text_vocab = length_rct_20k_text_vocab
        self.token_embed_OUTPUT_DIM = token_embed_OUTPUT_DIM
        self.token_embed_mask_zero = token_embed_mask_zero
        self.LOSS = LOSS
        self.METRICS = METRICS
        self.ACTIVATION = ACTIVATION
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
        #self.src_file_path = os.path.join(os.getcwd(),ZIP_FILE_NAME)
        self.BEST_MODEL_PATH = os.path.join(self.MODEL_EVALUATION_MODEL_DIR,BEST_MODEL_DIR) #TODO Make sure correct. Change back to os.getcwd() if error


@dataclass
class ModelPusherConfig:

    def __init__(self):
        self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, MODEL_NAME)
        self.MODEL_NAME = MODEL_NAME
        self.BEST_MODEL_PATH = os.path.join(os.getcwd(),BEST_MODEL_DIR)
        self.BEST_MODEL_TEXT_VECTORIZER_DIR_PATH_SOURCE: str = os.path.join(os.getcwd(),BEST_MODEL_DIR,TEXT_VECTORIZER_NAME)
        self.BEST_MODEL_TOKEN_EMBED_DIR_PATH_SOURCE: str = os.path.join(os.getcwd(),BEST_MODEL_DIR,TOKEN_EMBED_NAME)
        
