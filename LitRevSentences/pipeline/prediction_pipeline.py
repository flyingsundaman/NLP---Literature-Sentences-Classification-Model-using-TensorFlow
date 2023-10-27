import os
import io
import sys
import keras
import pickle
import shutil
from PIL import Image
from LitRevSentences.logger import logging
from LitRevSentences.constants import *
from LitRevSentences.exception import CustomException
from keras.utils import pad_sequences
#from hate.configuration.gcloud_syncer import GCloudSync
from LitRevSentences.components.data_transformation import DataTransformation
from LitRevSentences.entity.config_entity import DataTransformationConfig
from LitRevSentences.entity.artifacts_entity import DataIngestionArtifacts, ModelPusherArtifacts

class PredictionPipeline:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.best_model_path = BEST_MODEL_DIR # ModelPusherArtifacts.best_model_path
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig,data_ingestion_artifacts=DataIngestionArtifacts)

    """def copy_file_locally(self,best_model_dir_path) -> None:
        try:
            shutil.copy(best_model_dir_path, self.model_path)
        except Exception as e:
            raise CustomException(e, sys) from e"""

    def get_model_from_gcloud(self, Model_pusher_artifacts: ModelPusherArtifacts) -> str:
        """
        Method Name :   get_model_from_gcloud
        Description :   This method to get best model from google cloud storage
        Output      :   best_model_path
        """
        logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")
        try:
            # Loading the best model from s3 bucket
            #os.makedirs(self.model_path, exist_ok=True)
            #self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)
            #best_model_dir_path = os.path.join(os.getcwd(), self.best_model_path, self.model_name)
            #self.copy_file_locally(best_model_dir_path)
            #best_model_path = os.path.join(self.model_path, self.model_name)
            best_model_path = Model_pusher_artifacts.best_model_path
            logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    
    def predict(self,best_model_path,text):
        """load image, returns cuda tensor"""
        logging.info("Running the predict function")
        try:
            best_model_path:str = self.get_model_from_gcloud()
            load_model=keras.models.load_model(best_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            
            text=self.data_transformation.NLP_data_cleaning(text)
            text = [text]            
            print(text)
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)
            print(seq)

            
            model_pred_probs = load_model.predict(test_dataset)
            model_preds = tf.argmax(model_pred_probs, axis=1)



            pred = load_model.predict(padded)
            pred
            print("pred", pred)
            if pred>0.3:

                print("hate and abusive")
                return "hate and abusive"
            else:
                print("no hate")
                return "no hate"
        except Exception as e:
            raise CustomException(e, sys) from e

    
    def run_pipeline(self,text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:

            best_model_path: str = self.get_model_from_gcloud() 
            predicted_text = self.predict(best_model_path,text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e

