import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from LitRevSentences.logger import logging 
from LitRevSentences.exception import CustomException
from LitRevSentences.entity.config_entity import DataTransformationConfig
from LitRevSentences.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    # Function to convert the text file into a list of dictionaries
    def preprocess_text_with_line_numbers(self, file_paths:list):
        try:
            logging.info("Entered into the preprocess_text_with_line_numbers function")
            list_dicts = []
            for file_path in file_paths:
                with open(file_path, "r") as f:
                    input_lines = f.readlines()
                list_dict = []
                for i in input_lines:
                    if i.startswith("###"):
                        id = i[:-1]
                    elif i.isspace():
                        pass
                    else:
                        line = i.split("\t")
                        list_dict.append({"id":id,"target":line[0],"text":line[1]})
                list_dicts.append(list_dict)
            logging.info("Exited the preprocess_text_with_line_numbers function")
            return list_dicts
        except Exception as e:
            raise CustomException(e,sys) from e

    def One_hot_encode_labels(self, dfs:list):
        try:
            logging.info("Entered into the One_hot_encode_labels function")
            count = 0
            labels_one_hot_list = []
            for df in dfs:
                if count == 0:
                    logging.info(f"declare one_hot_encoder and fit it with the df")
                    one_hot_encoder = OneHotEncoder(sparse_output=False)
                    labels_one_hot = one_hot_encoder.fit_transform(df["target"].to_numpy().reshape(-1, 1))
                    count += 1
                    labels_one_hot_list.append(labels_one_hot)
                else:
                    logging.info(f"transform the df with one_hot_encoder")
                    labels_one_hot = one_hot_encoder.transform(df["target"].to_numpy().reshape(-1, 1))
                    labels_one_hot_list.append(labels_one_hot)
            
            logging.info("Exited the One_hot_encode_labels function")
            return labels_one_hot_list[0] , labels_one_hot_list[1], labels_one_hot_list[2]  
        except UnboundLocalError:
            
            logging.info("Exited the One_hot_encode_labels function")
            return labels_one_hot    

    def label_encoder(self, dfs:list):

        try:
            logging.info("Entered into the label_encoder function")
            count = 0
            label_encoder_list = []
            for df in dfs:
                if count == 0:
                    logging.info(f"declare label_encoder and fit it with the df")
                    label_encoder = LabelEncoder()
                    encoder = label_encoder.fit_transform(df["target"].to_numpy())
                    count += 1
                    label_encoder_list.append(encoder)
                else:
                    logging.info(f"transform the df with label_encoder")
                    encoder = label_encoder.transform(df["target"].to_numpy())
                    label_encoder_list.append(encoder)

            logging.info("Exited the label_encoder function")
            return label_encoder_list
        except Exception as e:
            raise CustomException(e,sys) from e

    def raw_data_cleaning(self, file_paths:list):
        
        try:
            logging.info("Entered into the raw_data_cleaning function")
            list_dicts = self.preprocess_text_with_line_numbers(file_paths=file_paths)

            df_train = pd.DataFrame(list_dicts[0])
            df_test = pd.DataFrame(list_dicts[1])
            df_val = pd.DataFrame(list_dicts[2])

            # Get lists of sentences
            train_sentences = df_train.text.tolist()
            test_sentences = df_test.text.tolist()
            val_sentences = df_val.text.tolist()

            # One hot encode for labels
            train_labels_one_hot, test_labels_one_hot, val_labels_one_hot = self.One_hot_encode_labels([df_train,df_test,df_val])

            train_labels_encoded, test_labels_encoded, val_labels_encoded = self.label_encoder([df_train,df_test,df_val])

            logging.info(f"Exited the raw_data_cleaning function")
            return train_sentences, test_sentences, val_sentences, train_labels_one_hot, test_labels_one_hot, val_labels_one_hot, train_labels_encoded, test_labels_encoded, val_labels_encoded
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def Create_Prefetch_Dataset(self, sentences, labels_one_hot):
        try:
            logging.info("Entered into the Create_Prefetch_Dataset function")
            Prefetch_dataset = tf.data.Dataset.from_tensor_slices((sentences, labels_one_hot))
            Prefetch_dataset = Prefetch_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
            logging.info("Exited the Create_Prefetch_Dataset function")
            return Prefetch_dataset
        except Exception as e:
            raise CustomException(e, sys) from e            

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)

            #train_sentences, train_labels_one_hot = self.raw_data_cleaning(self.data_ingestion_artifacts.train_data_file_path)
            #test_sentences, test_labels_one_hot = self.raw_data_cleaning(self.data_ingestion_artifacts.test_data_file_path)
            #val_sentences, val_labels_one_hot = self.raw_data_cleaning(self.data_ingestion_artifacts.val_data_file_path)
            #df[self.data_transformation_config.TWEET]=df[self.data_transformation_config.TWEET].apply(self.NLP_data_cleaning)
            dataset_list = [self.data_ingestion_artifacts.train_data_file_path , self.data_ingestion_artifacts.test_data_file_path, self.data_ingestion_artifacts.val_data_file_path]
            train_sentences, test_sentences, val_sentences, train_labels_one_hot, test_labels_one_hot, val_labels_one_hot, train_labels_encoded, test_labels_encoded, val_labels_encoded = self.raw_data_cleaning(dataset_list)

            """with open(self.data_transformation_config.TRAIN_SENTENCES_FILE_PATH, "w") as file:
                for item in train_sentences:
                    file.write(str(item) + "\n")"""
            

            pd.DataFrame(train_sentences).to_csv(self.data_transformation_config.TRAIN_SENTENCES_FILE_PATH,index=False,header=False)
            pd.DataFrame(test_sentences).to_csv(self.data_transformation_config.TEST_SENTENCES_FILE_PATH,index=False,header=False)
            pd.DataFrame(val_sentences).to_csv(self.data_transformation_config.VAL_SENTENCES_FILE_PATH,index=False,header=False)
            pd.DataFrame(train_labels_one_hot).to_csv(self.data_transformation_config.TRAIN_ONE_HOT_FILE_PATH,index=False,header=False)
            pd.DataFrame(test_labels_one_hot).to_csv(self.data_transformation_config.TEST_ONE_HOT_FILE_PATH,index=False,header=False)
            pd.DataFrame(val_labels_one_hot).to_csv(self.data_transformation_config.VAL_ONE_HOT_FILE_PATH,index=False,header=False)

            self.data_transformation_config.TRAIN_DATASET = self.Create_Prefetch_Dataset(train_sentences, train_labels_one_hot)
            self.data_transformation_config.TEST_DATASET = self.Create_Prefetch_Dataset(test_sentences, test_labels_one_hot)
            self.data_transformation_config.VAL_DATASET = self.Create_Prefetch_Dataset(val_sentences, val_labels_one_hot)

            data_transformation_artifact = DataTransformationArtifacts(
                train_dataset= self.data_transformation_config.TRAIN_DATASET,
                test_dataset= self.data_transformation_config.TEST_DATASET,
                val_dataset= self.data_transformation_config.VAL_DATASET,

                train_sentences = train_sentences,
                test_sentences=test_sentences,
                val_sentences=val_sentences,

                train_labels_one_hot = train_labels_one_hot,
                test_labels_one_hot = test_labels_one_hot,
                val_labels_one_hot = val_labels_one_hot,

                train_labels_encoded = train_labels_encoded,  
                test_labels_encoded = test_labels_encoded,
                val_labels_encoded = val_labels_encoded
            )
            logging.info("returning the DataTransformationArtifacts")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    




