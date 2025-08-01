import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from utils.tokenizer import clean_sentence, tokenize, build_vocab
from utils.dataset import TranslationDataset, collate_fn
from models.encoder import Encoder
from models.decoder import Decoder
import config


def load_dataset(path):
    eng_sentences, nep_sentences = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eng_sentences.append(clean_sentence(parts[0]))
                nep_sentences.append(clean_sentence(parts[1]))
    return eng_sentences, nep_sentences


def main():
    eng_raw, nep_raw = load_dataset("data/eng-npi.txt")
    eng_tokens = [tokenize(s) for s in eng_raw]
    nep_tokens = [tokenize(s) for s in nep_raw]

    eng_vocab = build_vocab(eng_tokens)
    nep_vocab = build_vocab(nep_tokens)

    #  Dataset & DataLoader
    train_data = TranslationDataset(nep_tokens, eng_tokens, nep_vocab, eng_vocab)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    #  Models on device (CPU or GPU)
    encoder = Encoder(len(nep_vocab), config.EMB_DIM, config.HIDDEN_DIM).to(config.DEVICE)
    decoder = Decoder(len(eng_vocab), config.EMB_DIM, config.HIDDEN_DIM).to(config.DEVICE)

    #  Optimizer & Loss
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses = []  #  store epoch losses

    print(f" Training on: {config.DEVICE}")
    for epoch in range(config.EPOCHS):
        encoder.train()
        decoder.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for src, trg in progress_bar:
            src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)

            optimizer.zero_grad()
            hidden, cell = encoder(src)

            input_token = trg[0, :]
            loss = torch.tensor(0.0, device=config.DEVICE)

            #  Teacher forcing
            for t in range(1, trg.shape[0]):
                output, hidden, cell = decoder(input_token, hidden, cell)
                loss += criterion(output, trg[t])
                input_token = trg[t]

            #  Backpropagation
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() / (trg.shape[0] - 1)
            total_loss += batch_loss
            progress_bar.set_postfix({"Batch Loss": f"{batch_loss:.4f}"})

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f" Epoch {epoch+1}/{config.EPOCHS} | Loss: {avg_loss:.4f}")

    #  Save the models & vocabularies for later use in translate.py
    os.makedirs("saved_models", exist_ok=True)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "eng_vocab": eng_vocab,
        "nep_vocab": nep_vocab
    }, "saved_models/translation_model.pth")

    print(" Model & vocabularies saved to saved_models/translation_model.pth")

    #  Plot the loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, config.EPOCHS + 1), losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid()
    plt.savefig("training_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
