import config
from datasets import load_dataset
import pandas as pd
class DataGen:

    def __init__(self, data_id, split):
        self.data_id = data_id
        self.split = split

    def load_req_data(self):
        data = load_dataset(self.data_id, config.SRC_LANG_TGT_LANG, split=self.split)
        self.src_data = data
        self.tgt_data = data

    def src_tgt_combine_data(self):
        self.src_data.set_format(type="pandas")
        src_df = self.src_data[config.REQ_DATA_COLUMN_SRC][:]
        self.tgt_data.set_format(type="pandas")
        tgt_df = self.tgt_data[config.REQ_DATA_COLUMN_TGT][:]
        data_df = pd.concat([src_df, tgt_df], axis=1)
        return data_df 
    
    def run(self):
        self.load_req_data()
        translation_data = self.src_tgt_combine_data()
        return translation_data

if __name__ == "__main__":
    test = DataGen(config.DATASET_ID, "test")
    test_data = test.run()
    test_df = pd.DataFrame(test_data)

    valid = DataGen(config.DATASET_ID, "validation")
    valid_data = valid.run()
    valid_df = pd.DataFrame(valid_data)

    trans_df = pd.concat([test_df, valid_df], axis=0).reset_index(drop=True)
    trans_df.columns = ["eng", "hin"]
    trans_df.to_csv(config.DATA_GEN_PATH, index=False)
    print(trans_df)
