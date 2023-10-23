import os
import re
import sys
import string
import pandas as pd
#import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from LitRevSentences.logger import logging 
from LitRevSentences.exception import CustomException
from LitRevSentences.entity.config_entity import DataTransformationConfig
from LitRevSentences.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def raw_data_cleaning(self):
        
        try:
            logging.info("Entered into the raw_data_cleaning function")
            df = pd.read_csv(self.data_ingestion_artifacts.data_file_path)
            df.drop(self.data_transformation_config.DROP_COLUMNS,axis = self.data_transformation_config.AXIS,
            inplace = self.data_transformation_config.INPLACE)

            df[df[self.data_transformation_config.CLASS]==0][self.data_transformation_config.CLASS]=1
            
            # replace the value of 0 to 1
            df[self.data_transformation_config.CLASS].replace({0:1},inplace=True)

            # Let's replace the value of 2 to 0.
            df[self.data_transformation_config.CLASS].replace({2:0}, inplace = True)

            # Let's change the name of the 'class' to label
            df.rename(columns={self.data_transformation_config.CLASS:self.data_transformation_config.LABEL},inplace =True)
            df['tweet'] = df['tweet'].astype(str)
            logging.info(f"Exited the raw_data_cleaning function and returned the raw_data {df}")
            return df

        except Exception as e:
            raise CustomException(e,sys) from e

    def NLP_data_cleaning(self, words):

        try:
            logging.info("Entered into the NLP_data_cleaning function")
            # Let's apply stemming and stopwords on the data
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            words = str(words).lower()
            words = re.sub('\[.*?\]', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = [word for word in words.split(' ') if words not in stopword]
            words=" ".join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words=" ".join(words)
            logging.info("Exited the NLP_data_cleaning function")
            return words 

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")
            #self.imbalance_data_cleaning()
            df = self.raw_data_cleaning()
            #df = self.concat_dataframe()
            df[self.data_transformation_config.TWEET]=df[self.data_transformation_config.TWEET].apply(self.NLP_data_cleaning)

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH,index=False,header=True)

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_data_path = self.data_transformation_config.TRANSFORMED_FILE_PATH
            )
            logging.info("returning the DataTransformationArtifacts")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    




