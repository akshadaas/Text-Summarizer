import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import torch
from src.textSummarizer.logging import logger
from src.textSummarizer.config.configurations import ModelTrainerConfig
from datasets import load_from_disk

class ModelTrainer:
    def __init__(self,config = ModelTrainerConfig):
        self.config = config


    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        trainer_args = TrainingArguments(
                output_dir = self.config.root_dir, 
                num_train_epochs = self.config.num_train_epochs,
                warmup_steps =  self.config.warmup_steps,
                per_device_train_batch_size =  self.config.per_device_train_batch_size,
                per_device_eval_batch_size =  self.config.per_device_eval_batch_size,
                weight_decay =  self.config.weight_decay,
                logging_steps =  self.config.logging_steps,
                evaluation_strategy =  self.config.evaluation_strategy, #evaluation performed after  a specified #steps in eval_steps
                eval_steps =  self.config.eval_steps,
                save_steps =  self.config.save_steps,
                gradient_accumulation_steps =  self.config.gradient_accumulation_steps

                )

        trainer =Trainer(
                model=model,
                args =trainer_args,
                tokenizer =tokenizer,
                data_collator =seq2seq_data_collator,
                train_dataset= dataset_samsum_pt['test'],
                eval_dataset = dataset_samsum_pt['validation'])

        print('running.........')
        trainer.train()

        model.save_pretrained(os.path.join(self.config.root_dir,'pegasus-samsum-model'))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,'pegasus-samsum-tokenizer'))
