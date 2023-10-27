from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    train_data_file_path: str
    test_data_file_path: str
    val_data_file_path: str

@dataclass
class DataTransformationArtifacts:
    train_dataset: str
    test_dataset: str
    val_dataset: str
    train_sentences: list
    test_sentences: list
    val_sentences: list
    train_labels_one_hot: list
    test_labels_one_hot: list
    val_labels_one_hot: list
    train_labels_encoded: list
    test_labels_encoded: list
    val_labels_encoded: list
    label_encoder: object 

@dataclass
class ModelTrainerArtifacts: 
    trained_model_path: str
    text_vectorizer: object
    token_embed: object

@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool    


@dataclass
class ModelPusherArtifacts:
    best_model_path: str
    text_vectorizer: str
    token_embed: str
    label_encoder: str