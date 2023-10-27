import os 
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
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

    """
    def spliting_data(self,csv_path):
        try:
            logging.info("Entered the spliting_data function")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data into x and y")
            x = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train_test_split on the data")
            x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 42)
            print(len(x_train),len(y_train))
            print(len(x_test),len(y_test))
            print(type(x_train),type(y_train))
            logging.info("Exited the spliting the data function")
            return x_train,x_test,y_train,y_test

        except Exception as e:
            raise CustomException(e, sys) from e"""
    
    """
    def tokenizing(self,x_train):
        try:
            logging.info("Applying tokenization on the data")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences,maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f" The sequence matrix is: {sequences_matrix}")
            return sequences_matrix,tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e"""

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
            #x_train,x_test,y_train,y_test = self.spliting_data(csv_path=self.data_transformation_artifacts.transformed_data_path)
            train_dataset, test_dataset, val_dataset = self.data_transformation_artifacts.train_dataset, self.data_transformation_artifacts.test_dataset, self.data_transformation_artifacts.val_dataset
            model_architecture = ModelArchitecture()   

            text_vectorizer = self.text_vectorizer_fn(train_sentences = self.data_transformation_artifacts.train_sentences)

            token_embed = self.token_embedding_layer_fn()

            model = model_architecture.get_model(text_vectorizer=text_vectorizer, token_embed=token_embed)

            #logging.info(f"Xtrain size is : {x_train.shape}")

            #logging.info(f"Xtest size is : {x_test.shape}")

            #sequences_matrix,tokenizer =self.tokenizing(x_train)
            

            logging.info("Entered into model training")
            """model.fit(sequences_matrix, y_train, 
                        batch_size=self.model_trainer_config.BATCH_SIZE, 
                        epochs = self.model_trainer_config.EPOCH, 
                        validation_split=self.model_trainer_config.VALIDATION_SPLIT, 
                        )"""


            model.fit(train_dataset,
                    steps_per_epoch=int(0.1 * len(train_dataset)),
                    epochs=self.model_trainer_config.EPOCH,
                    validation_data=val_dataset,
                    validation_steps=int(0.1 * len(val_dataset)))


            logging.info("Model training finished")
            """with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)"""
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)

            logging.info("saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            #x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            #y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            #x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                text_vectorizer = text_vectorizer,
                token_embed = token_embed
                #test_dataset = test_dataset
                #y_test_path = self.model_trainer_config.Y_TEST_DATA_PATH
                )
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

