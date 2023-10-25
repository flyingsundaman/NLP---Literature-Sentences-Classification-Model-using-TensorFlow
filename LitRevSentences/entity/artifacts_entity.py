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

@dataclass
class ModelTrainerArtifacts: 
    trained_model_path: str
    #x_test_path: list
    #y_test_path: list

@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool    


@dataclass
class ModelPusherArtifacts:
    best_model_path: str