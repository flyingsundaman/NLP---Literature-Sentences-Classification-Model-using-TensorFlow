import sys
import os
import shutil
import pickle
import joblib
import tensorflow as tf
from LitRevSentences.logger import logging
from LitRevSentences.exception import CustomException
from LitRevSentences.entity.config_entity import ModelPusherConfig
from LitRevSentences.entity.artifacts_entity import ModelPusherArtifacts, ModelEvaluationArtifacts, ModelTrainerArtifacts

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifacts: ModelEvaluationArtifacts, model_trainer_artifacts: ModelTrainerArtifacts):
        """
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_pusher_config = model_pusher_config
        self.model_evaluation_artifacts = model_evaluation_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
    def copy_file_locally(self) -> None:
        shutil.copy(self.model_pusher_config.TRAINED_MODEL_PATH, self.model_pusher_config.BEST_MODEL_PATH)

    def save_tensor_config_weights(tensor, filepath):
        try:
            # Serialize the text vectorizer's configuration and weights
            config = tensor.get_config()
            weights = tensor.get_weights()
            
            with open(filepath, 'wb') as handle:
                tf.saved_model.save((config, weights), handle)
        except Exception as e:
            raise CustomException(e, sys) from e
            
    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
            Method Name :   initiate_model_pusher
            Description :   This method initiates model pusher.

            Output      :    Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")
        try:
            # Uploading the model to gcloud storage
            os.makedirs(self.model_pusher_config.BEST_MODEL_PATH, exist_ok=True)
            
            self.copy_file_locally()
            """with open('text_vectorizer.pickle', 'wb') as handle:
                pickle.dump(self.model_trainer_artifacts.text_vectorizer , handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('token_embed.pickle', 'wb') as handle:
                pickle.dump(self.model_trainer_artifacts.token_embed , handle, protocol=pickle.HIGHEST_PROTOCOL)"""
            
            # Save text_vectorizer
            #joblib.dump(self.model_trainer_artifacts.text_vectorizer, self.model_pusher_config.BEST_MODEL_TEXT_VECTORIZER_DIR_PATH_SOURCE)
            """self.save_tensor_config_weights(tensor = self.model_trainer_artifacts.text_vectorizer,
                                      filepath = self.model_pusher_config.BEST_MODEL_TEXT_VECTORIZER_DIR_PATH_SOURCE)"""
            tf.saved_model.save(self.model_trainer_artifacts.text_vectorizer, self.model_pusher_config.BEST_MODEL_TEXT_VECTORIZER_DIR_PATH_SOURCE)

            # Save token_embed
            #joblib.dump(self.model_trainer_artifacts.token_embed, self.model_pusher_config.BEST_MODEL_TOKEN_EMBED_DIR_PATH_SOURCE)
            """self.save_tensor_config_weights(tensor = self.model_trainer_artifacts.token_embed,
                                      filepath = self.model_pusher_config.BEST_MODEL_TOKEN_EMBED_DIR_PATH_SOURCE)"""
            tf.saved_model.save(self.model_trainer_artifacts.token_embed, self.model_pusher_config.BEST_MODEL_TOKEN_EMBED_DIR_PATH_SOURCE)

            logging.info("Uploaded best model to best_Model_saved folder")

            # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifacts(
                best_model_path=self.model_pusher_config.BEST_MODEL_PATH
            )
            logging.info("Exited the initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
