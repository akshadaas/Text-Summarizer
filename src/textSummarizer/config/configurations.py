from src.textSummarizer.constants import *
from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.entity import (DataIngestionConfig,DataValidationConfig, 
DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig)

class ConfigurationManager:

    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_Path):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_roots])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        self.config = self.config.data_ingestion

        create_directories([self.config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = self.config.root_dir,
            source_url = self.config.source_url,
            local_data_file = self.config.local_data_file,
            unzip_dir = self.config.unzip_dir
        )    


        return data_ingestion_config



    def get_data_validation_config(self) -> DataValidationConfig:
        self.config = self.config.data_validation

        create_directories([self.config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir = self.config.root_dir,
            STATUS_FILE =  self.config.STATUS_FILE,
            ALL_REQUIRED_FILES =  self.config.ALL_REQUIRED_FILES
        )    


        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        self.config = self.config.data_transformation

        create_directories([self.config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = self.config.root_dir,
            data_path =  self.config.data_path,
            tokenizer_name =  self.config.tokenizer_name
        )    


        return data_transformation_config


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        self.config = self.config.model_trainer
        self.params = self.params.TrainingArguments

        create_directories([self.config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir = self.config.root_dir,
            data_path =  self.config.data_path,
            model_ckpt =  self.config.model_ckpt,
            num_train_epochs  = self.params.num_train_epochs,
            warmup_steps  = self.params.warmup_steps,
            per_device_train_batch_size = self.params.per_device_train_batch_size,
            per_device_eval_batch_size = self.params.per_device_eval_batch_size,
            weight_decay = self.params.weight_decay,
            logging_steps = self.params.logging_steps,
            evaluation_strategy = self.params.evaluation_strategy,
            eval_steps = self.params.eval_steps,
            save_steps  = self.params.save_steps,
            gradient_accumulation_steps = self.params.gradient_accumulation_steps
        )    


        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        self.config = self.config.model_evaluation

        create_directories([self.config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = self.config.root_dir,
            data_path =  self.config.data_path,
            model_path = self.config.model_path,
            tokenizer_path =  self.config.tokenizer_path,
            metric_file_name = self.config.model_file_name
        )    


        return model_evaluation_config
