# utils/tokenizer.py

import re
from collections import Counter
from config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

def clean_sentence(sent):
    sent = sent.lower().strip()
    # Keeps: English letters, Devanagari characters, digits (Latin and Nepali), whitespace
    return re.sub(r"[^\wँ-ॿ०-९\s]", "", sent)

def tokenize(sentence):
    sentence = clean_sentence(sentence)
    return sentence.strip().split()

def build_vocab(token_lists, min_freq=1):
    counter = Counter(token for tokens in token_lists for token in tokens)
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def sentence_to_ids(tokens, vocab):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

def wrap_sos_eos(ids, vocab):
    return [vocab[SOS_TOKEN]] + ids + [vocab[EOS_TOKEN]]

def ids_to_sentence(ids, reverse_vocab):
    tokens = [reverse_vocab[i] for i in ids if i != reverse_vocab[PAD_TOKEN]]
    return " ".join(tokens)

def batch_to_ids(token_batch, vocab):
    return [sentence_to_ids(tokens, vocab) for tokens in token_batch]
