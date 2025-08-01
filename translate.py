import torch
from utils.tokenizer import tokenize
from models.encoder import Encoder
from models.decoder import Decoder
import config

#  Load model checkpoint
checkpoint = torch.load("saved_models/translation_model.pth", map_location=config.DEVICE)

eng_vocab = checkpoint["eng_vocab"]
nep_vocab = checkpoint["nep_vocab"]

#  Build index-to-word mapping for English vocab
idx2word = {idx: word for word, idx in eng_vocab.items()}

#  Initialize models
encoder = Encoder(len(nep_vocab), config.EMB_DIM, config.HIDDEN_DIM).to(config.DEVICE)
decoder = Decoder(len(eng_vocab), config.EMB_DIM, config.HIDDEN_DIM).to(config.DEVICE)

encoder.load_state_dict(checkpoint["encoder_state"])
decoder.load_state_dict(checkpoint["decoder_state"])

encoder.eval()
decoder.eval()

def translate_sentence(sentence, max_length=20):
    """Translate a Nepali sentence to English using greedy decoding."""
    tokens = tokenize(sentence)
    input_indices = [nep_vocab.get(token, nep_vocab.get("<unk>", 0)) for token in tokens]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(1).to(config.DEVICE)

    with torch.no_grad():
        hidden, cell = encoder(input_tensor)

    input_token = torch.tensor([eng_vocab["<sos>"]], dtype=torch.long).to(config.DEVICE)

    outputs = []
    with torch.no_grad():
        for _ in range(max_length):
            output, hidden, cell = decoder(input_token, hidden, cell)
            predicted_id = output.argmax(1).item()

            if predicted_id == eng_vocab["<eos>"]:
                break

            outputs.append(predicted_id)
            input_token = torch.tensor([predicted_id], dtype=torch.long).to(config.DEVICE)

    translated_words = [idx2word[idx] for idx in outputs if idx in idx2word]
    return " ".join(translated_words)


if __name__ == "__main__":
    input_file = "sentences.txt"      #  Nepali sentences input
    output_file = "translations.txt"  # Translations output

    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            translation = translate_sentence(sentence)
            f.write(f"{sentence} ➡ {translation}\n")

    print(f" Translations saved to {output_file}")
