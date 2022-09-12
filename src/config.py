from unittest.util import _MAX_LENGTH


DATASET_ID  = "Helsinki-NLP/tatoeba_mt"
SRC_LANG_TGT_LANG = "eng-hin"
REQ_DATA_COLUMN_SRC = "sourceString"
REQ_DATA_COLUMN_TGT = "targetString"
DATA_GEN_PATH = "artifacts/translation.csv"
TEST_RATIO = 0.2
RANDOM_STATE = 0

MODEL_CKPT = "Helsinki-NLP/opus-mt-en-hi"
MAX_LENGTH = 128

MODEL_OUT_NAME = "eng-hin-translator"
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE =  16
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5

