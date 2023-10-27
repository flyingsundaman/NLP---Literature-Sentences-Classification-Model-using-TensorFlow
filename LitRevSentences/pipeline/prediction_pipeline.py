import os
import sys
import keras
import pickle
import pandas as pd
import tensorflow as tf
import nltk
import re
#nltk.download('punkt')
from LitRevSentences.logger import logging
from LitRevSentences.constants import *
from LitRevSentences.exception import CustomException
from LitRevSentences.components.data_transformation import DataTransformation
from LitRevSentences.entity.config_entity import DataTransformationConfig
from LitRevSentences.entity.artifacts_entity import DataIngestionArtifacts

class PredictionPipeline:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.best_model_path = BEST_MODEL_DIR # ModelPusherArtifacts.best_model_path
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig,data_ingestion_artifacts=DataIngestionArtifacts)

    def get_best_model_artifacts(self) -> str:
        """
        Method Name :   get_best_model_artifacts
        Description :   This method to get best model from google cloud storage
        Output      :   best_model_path
        """
        logging.info("Entered the get_best_model_artifacts method of PredictionPipeline class")
        try:

            best_model_path = os.path.join(os.getcwd(),BEST_MODEL_DIR, MODEL_NAME)  #Model_pusher_artifacts.best_model_path
            #text_vectorizer = os.path.join(os.getcwd(),BEST_MODEL_DIR,TEXT_VECTORIZER_NAME)
            #token_embed = os.path.join(os.getcwd(),BEST_MODEL_DIR,TOKEN_EMBED_NAME)
            label_encoder = os.path.join(os.getcwd(),BEST_MODEL_DIR,LABEL_ENCODER_NAME)
            logging.info("Exited the get_best_model_artifacts method of PredictionPipeline class")
            return best_model_path, label_encoder

        except Exception as e:
            raise CustomException(e, sys) from e

    def add_space_word_comma_dot(self,LitRev):
        try:
            LitRev = re.sub(r'(?<=[\w])([.,])', r' \1', LitRev)
            return LitRev
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def replace_numbers_with_at(self,LitRev):
        try:
            return re.sub(r'\d+', '@', LitRev)
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def text_preprocess_predict(self,LitRev):
        try:
            # Convert the Literature review into a list of sentences
            sentences = nltk.tokenize.sent_tokenize(LitRev)
            sentences_raw = sentences.copy()
            sentences = [self.add_space_word_comma_dot(LitRev=sentence) for sentence in sentences]
            #LitRev = self.add_space_word_comma_dot(LitRev=LitRev)

            sentences = [self.replace_numbers_with_at(LitRev=sentence) for sentence in sentences]
            #LitRev = self.replace_numbers_with_at(LitRev=LitRev)

            # Convert the Literature review into a list of sentences
            #sentences = nltk.tokenize.sent_tokenize(LitRev)
            return sentences, sentences_raw
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def predict(self,LitRev):
        """load image, returns cuda tensor"""
        logging.info("Running the predict function")
        try:
            best_model_path, label_encoder  = self.get_best_model_artifacts()
            load_model=keras.models.load_model(best_model_path)
            #text_vectorizer = tf.saved_model.load(text_vectorizer)
            #token_embed = tf.saved_model.load(token_embed)
            with open(label_encoder, 'rb') as handle:
                label_encoder = pickle.load(handle)
            sentences_list, sentences_raw = self.text_preprocess_predict(LitRev=LitRev)
            
            model_pred_probs = load_model.predict(sentences_list)
            model_preds = tf.argmax(model_pred_probs, axis=1)
            # Get list of class names of test predictions
            test_pred_classes = [label_encoder.classes_[pred] for pred in model_preds]

            result = pd.DataFrame({'Category': test_pred_classes, 'Sentence': sentences_raw})

            print(result)

            return result
        except Exception as e:
            raise CustomException(e, sys) from e

    
    def run_pipeline(self,LitRev):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            result_df = self.predict(LitRev=LitRev)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return result_df
        except Exception as e:
            raise CustomException(e, sys) from e

