from src.textSummarizer.config.configurations import ConfigurationManager
from src.textSummarizer.components.model_evaluation import ModelEvaluation
from src.textSummarizer.logging import logger


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):

        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluate = DataTransformation(config = model_evaluation_config)
        model_evaluate.evaluate()
