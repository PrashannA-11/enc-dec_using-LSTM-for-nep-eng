# utils/dataset.py

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.tokenizer import sentence_to_ids, wrap_sos_eos
from config import PAD_TOKEN  # to get correct padding index

class TranslationDataset(Dataset):
    def __init__(self, src_token_lists, trg_token_lists, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.src_data = [
            wrap_sos_eos(sentence_to_ids(tokens, src_vocab), src_vocab)
            for tokens in src_token_lists
        ]
        self.trg_data = [
            wrap_sos_eos(sentence_to_ids(tokens, trg_vocab), trg_vocab)
            for tokens in trg_token_lists
        ]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_tensor = torch.tensor(self.src_data[idx], dtype=torch.long)
        trg_tensor = torch.tensor(self.trg_data[idx], dtype=torch.long)
        return src_tensor, trg_tensor


def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    
    src_padded = pad_sequence(list(src_batch), padding_value=0)  # default 0 for now
    trg_padded = pad_sequence(list(trg_batch), padding_value=0)

    # Optional transpose for batch-first (depends on your model)
    return src_padded, trg_padded
