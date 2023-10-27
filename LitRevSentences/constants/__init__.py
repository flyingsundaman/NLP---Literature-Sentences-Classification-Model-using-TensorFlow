import os
from datetime import datetime

# Common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
#LABEL = 'label'
DATASET_DIR = "data"
TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
DEV_FILE = "dev.txt"
MODEL_NAME = 'model.keras'
TEXT_VECTORIZER_NAME =' text_vectorizer.pkl'
TOKEN_EMBED_NAME = 'token_embed.pkl'
#APP_HOST = "0.0.0.0"
#APP_PORT = 8080


# Data ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATASET_GitHub = "https://github.com/Franck-Dernoncourt/pubmed-rct/"
DATASET_NAME =  "PubMed_20k_RCT_numbers_replaced_with_at_sign"

# Data transformation constants 
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
TRAIN_SENTENCES_NAME = "train_sentences.csv"
TEST_SENTENCES_NAME = "test_sentences.csv"
VAL_SENTENCES_NAME = "val_sentences.csv"
TRAIN_ONE_HOT_NAME = "train_labels_one_hot.csv"
TEST_ONE_HOT_NAME = "test_labels_one_hot.csv"
VAL_ONE_HOT_NAME = "val_labels_one_hot.csv"

# Model training constants
MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
TRAINED_MODEL_DIR = 'trained_model'
MAX_TOKENS = 68000
OUTPUT_SEQUENCE_LENGTH = 55 
length_rct_20k_text_vocab = 64841
token_embed_OUTPUT_DIM = 128
token_embed_mask_zero = True
EPOCH = 0
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2


# Model Architecture constants
LOSS = "categorical_crossentropy"
METRICS = ['accuracy']
ACTIVATION = 'softmax'
OPTIMIZER = "adam"


# Model  Evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
BEST_MODEL_DIR = "best_Model_saved"
MODEL_EVALUATION_FILE_NAME = 'loss.csv'

# Model  pusher constants
#BEST_MODEL_DIR = os.path.join()
