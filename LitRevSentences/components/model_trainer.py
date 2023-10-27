import os 
import sys
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from LitRevSentences.logger import logging
from LitRevSentences.constants import *
from LitRevSentences.exception import CustomException
from LitRevSentences.model_architecture.model import ModelArchitecture
from LitRevSentences.entity.config_entity import ModelTrainerConfig
from LitRevSentences.entity.artifacts_entity import ModelTrainerArtifacts, DataTransformationArtifacts


class ModelTrainer:
    def __init__(self,data_transformation_artifacts: DataTransformationArtifacts,
                model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def text_vectorizer_fn(self,train_sentences):
        
        try:
            logging.info("Entered into the text_vectorizer_fn function")
            logging.info("Declare text_vectorizer and adapt")
            text_vectorizer = TextVectorization(max_tokens=self.model_trainer_config.MAX_TOKENS, output_sequence_length=self.model_trainer_config.OUTPUT_SEQUENCE_LENGTH)
            #Adapt text vectorizer to training sentences
            text_vectorizer.adapt(train_sentences)
            logging.info(f"Exited the text_vectorizer_fn function")
            return text_vectorizer
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def token_embedding_layer_fn(self):
        try:
            logging.info("Entered into the token_embedding_layer_fn function")
            token_embed = layers.Embedding(input_dim = self.model_trainer_config.length_rct_20k_text_vocab, #length of vocab
                                output_dim= self.model_trainer_config.token_embed_OUTPUT_DIM,
                                mask_zero= self.model_trainer_config.token_embed_mask_zero,
                                name="token_embedding")
            logging.info("exited the token_embedding_layer_fn function")
            return token_embed
        except Exception as e:
            raise CustomException(e,sys) from e    
    
    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered the initiate_model_trainer function ")
            train_dataset, test_dataset, val_dataset = self.data_transformation_artifacts.train_dataset, self.data_transformation_artifacts.test_dataset, self.data_transformation_artifacts.val_dataset
            model_architecture = ModelArchitecture()   

            text_vectorizer = self.text_vectorizer_fn(train_sentences = self.data_transformation_artifacts.train_sentences)

            token_embed = self.token_embedding_layer_fn()

            model = model_architecture.get_model(text_vectorizer=text_vectorizer, token_embed=token_embed)

            logging.info("Entered into model training")

            model.fit(train_dataset,
                    steps_per_epoch=int(0.1 * len(train_dataset)),
                    epochs=self.model_trainer_config.EPOCH,
                    validation_data=val_dataset,
                    validation_steps=int(0.1 * len(val_dataset)))

            logging.info("Model training finished")

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)

            logging.info("saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                text_vectorizer = text_vectorizer,
                token_embed = token_embed
                )
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e