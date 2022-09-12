import config
import numpy as np
import evaluate
import torch
import gc
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

class Model:

    def __init__(self, model_ckpt, train_data, eval_data):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
        self.train_data = train_data
        self.val_data = eval_data

    def compute_bleu_metric(self, eval_out):
        torch.cuda.empty_cache()
        gc.collect()
        metric = evaluate.load("sacrebleu")
        predictions, labels = eval_out

        if isinstance(predictions, tuple):
            predictions = predictions[0]
            
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_predictions = [pred.strip() for pred in decoded_predictions]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        bleu_score = metric.compute(predictions=decoded_predictions, references=decoded_labels)
        torch.cuda.empty_cache()
        gc.collect()
        return {"bleu_score": bleu_score["score"]}
    
    def training_args(self):
        
        logging_steps = len(self.train_data) // config.TRAIN_BATCH_SIZE

        args = Seq2SeqTrainingArguments(
                    output_dir = config.MODEL_OUT_NAME,
                    evaluation_strategy="epoch",
                    learning_rate=config.LEARNING_RATE,
                    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
                    per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
                    weight_decay=config.WEIGHT_DECAY,
                    num_train_epochs=config.NUM_EPOCHS,
                    save_total_limit=3,
                    predict_with_generate=True,
                    fp16=True,
                    logging_steps=logging_steps,
                    push_to_hub=True,
                )
        return args

    def train_model(self):
        trainer = Seq2SeqTrainer(
                    self.model,
                    self.training_args(),
                    train_dataset=self.train_data,
                    eval_dataset=self.val_data,
                    data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model),
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_bleu_metric
                )
        torch.cuda.empty_cache()
        gc.collect()
        trainer.train()
        trainer.push_to_hub()
        trainer.evaluate(max_length=config.MAX_LENGTH)


            