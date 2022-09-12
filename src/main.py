import config
import torch
import gc
from data import DatasetPrep
from model import Model

if __name__ == "__main__":
    dataprep = DatasetPrep(config.DATA_GEN_PATH, config.MODEL_CKPT)
    train_inp, val_inp = dataprep.run()
    model = Model(config.MODEL_CKPT, train_inp, val_inp)
    torch.cuda.empty_cache()
    gc.collect()
    model.train_model()
    