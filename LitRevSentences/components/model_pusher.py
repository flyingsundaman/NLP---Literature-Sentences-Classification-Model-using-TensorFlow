import sys
import os
import shutil
import pickle
import joblib
import tensorflow as tf
from LitRevSentences.logger import logging
from LitRevSentences.exception import CustomException
from LitRevSentences.entity.config_entity import ModelPusherConfig
from LitRevSentences.entity.artifacts_entity import ModelPusherArtifacts, ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifacts: ModelEvaluationArtifacts, model_trainer_artifacts: ModelTrainerArtifacts, data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_pusher_config = model_pusher_config
        self.model_evaluation_artifacts = model_evaluation_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
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
            
            # Save text_vectorizer
            tf.saved_model.save(self.model_trainer_artifacts.text_vectorizer, self.model_pusher_config.BEST_MODEL_TEXT_VECTORIZER_DIR_PATH_SOURCE)

            # Save token_embed
            tf.saved_model.save(self.model_trainer_artifacts.token_embed, self.model_pusher_config.BEST_MODEL_TOKEN_EMBED_DIR_PATH_SOURCE)

            with open(self.model_pusher_config.BEST_MODEL_LABEL_ENCODER_DIR_PATH_SOURCE, 'wb') as handle:
                pickle.dump(self.data_transformation_artifacts.label_encoder , handle, protocol=pickle.HIGHEST_PROTOCOL)

            logging.info("Uploaded best model to best_Model_saved folder")

            # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifacts(
                best_model_path=self.model_pusher_config.BEST_MODEL_PATH,
                text_vectorizer = self.model_pusher_config.BEST_MODEL_TEXT_VECTORIZER_DIR_PATH_SOURCE,
                token_embed = self.model_pusher_config.BEST_MODEL_TOKEN_EMBED_DIR_PATH_SOURCE,
                label_encoder = self.model_pusher_config.BEST_MODEL_LABEL_ENCODER_DIR_PATH_SOURCE
            )
            logging.info("Exited the initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
