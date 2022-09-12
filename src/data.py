import config
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class DatasetPrep:
    
    def __init__(self, data_path, model_ckpt):
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def load_data(self):
        self.df =  pd.read_csv(config.DATA_GEN_PATH)
        
    def split_data(self):
        train_df, valid_df = train_test_split(self.df, test_size=config.TEST_RATIO, random_state=config.RANDOM_STATE)
        self.train_df = train_df
        self.valid_df = valid_df
    
    def df_hf_data(self):
        train_data = Dataset.from_pandas(self.train_df)
        valid_data = Dataset.from_pandas(self.valid_df)
        return train_data, valid_data

    def preprocess_data(self, data):
        inputs = [datum for datum in data["eng"]]
        targets = [datum for datum in data["hin"]]
        model_inputs = self.tokenizer(inputs, max_length=config.MAX_LENGTH, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=config.MAX_LENGTH, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def run(self):
        self.load_data()
        print("Data Loaded")
        self.split_data()
        print("Data Splitted")
        train_data, val_data = self.df_hf_data()
        print("DF to HF done")
        train_model_inp = train_data.map(
                            self.preprocess_data,
                            batched=True,
                            remove_columns=train_data.column_names
                            )
        print("Train data tokenization done")
        val_model_inp = val_data.map(
                            self.preprocess_data,
                            batched=True,
                            remove_columns=val_data.column_names
                            )
        print("Validation data tokenization done")
        return train_model_inp, val_model_inp



    


