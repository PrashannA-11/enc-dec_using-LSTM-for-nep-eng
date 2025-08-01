# config.py

import torch

# Hyperparameters
EMB_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 64

# Special Tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
