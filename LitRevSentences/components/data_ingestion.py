import os
import sys
import shutil
import requests
from LitRevSentences.logger import logging
from LitRevSentences.exception import CustomException
from LitRevSentences.entity.config_entity import DataIngestionConfig
from LitRevSentences.entity.artifacts_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):

        """
        :param data_ingestion_config: Configuration for data ingestion
        """
        self.data_ingestion_config = data_ingestion_config

    def download_dataset_github(self) -> None:
        """
        Download the dataset from DATASET_GITHUB_LINK and save it to DATASET_PATH.

        Parameters:
        - DATASET_GITHUB_LINK (str): The GitHub URL of the dataset.
        - DATASET_PATH (str): The local path where you want to save the dataset.
        """        
        try:
            logging.info("Entered the download_dataset_github method of Data ingestion class")
            if os.path.exists(self.data_ingestion_config.DATASET_PATH):
                logging.info(f'The dataset already downloaded in previous iteration')
            else:
                response = requests.get(self.data_ingestion_config.DATASET_GITHUB_LINK)
                response.raise_for_status()  # Check for HTTP errors
                os.makedirs(self.data_ingestion_config.DATASET_PATH, exist_ok=True)
                with open(self.data_ingestion_config.DATASET_PATH, 'wb') as local_file:
                    local_file.write(response.content)
                logging.info(f"Dataset downloaded and saved to {self.data_ingestion_config.DATASET_PATH}")
            logging.info("Exited the download_dataset_github method of Data ingestion class")
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def copy_file_locally(self, sources: list, to: str) -> None:
        try:
            for source in sources:
                shutil.copy(source, to)
                logging.info(f'Dataset copied from {source} to {to}')
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_data_from_data_source(self) -> None:
        try:
            logging.info("Entered the get_data_from_data_source method of Data ingestion class")
            self.download_dataset_github()
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            dataset_list = [self.data_ingestion_config.TRAIN_DATA_PATH,self.data_ingestion_config.TEST_DATA_PATH,self.data_ingestion_config.VAL_DATA_PATH]
            self.copy_file_locally(dataset_list, self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
            logging.info("Exited the get_data_from_data_source method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Method Name :   initiate_data_ingestion
        Description :   This function initiates a data ingestion steps
        Output      :   Returns data ingestion artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")
        try:
            self.get_data_from_data_source()
            logging.info("Fetched the data from GitHub")
            train_data_file_path = self.data_ingestion_config.TRAIN_DATA_INGESTION_ARTIFACTS_DIR
            test_data_file_path = self.data_ingestion_config.TEST_DATA_INGESTION_ARTIFACTS_DIR
            val_data_file_path = self.data_ingestion_config.VAL_DATA_INGESTION_ARTIFACTS_DIR

            data_ingestion_artifacts = DataIngestionArtifacts(train_data_file_path=train_data_file_path,
                                                              test_data_file_path=test_data_file_path,
                                                              val_data_file_path=val_data_file_path)

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e