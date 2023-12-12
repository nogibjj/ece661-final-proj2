import torch

BATCH_SIZE = 64  # increase / decrease according to GPU memeory
NUM_EPOCHS = 100  # number of epochs to train for

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# classes: 0 index is reserved for background
CLASSES = ["Q1", "Q2", "Q3", "Q4"]
NUM_CLASSES = 4

# location to save model and plots
OUT_DIR = "checkpoints"

# model config
MAX_SEQ_LEN = 256
NHEAD = 8
D_HID = 512
NLAYERS = 6
EMSIZE = 256
DROPOUT = 0.1
VOCAB_SIZE = 198

END_TOKEN = 1
TEMPERATURE = 1.0

