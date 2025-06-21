# utils.py
def get_vocab(sentences):
    vocab = sorted(set("".join(sentences)))
    vocab = ["_"] + vocab  # "_" as the blank token at index 0
    char_to_index = {c: i for i, c in enumerate(vocab)}
    return vocab, char_to_index

def text_to_labels(text, char_to_index):
    return [char_to_index[c] for c in text if c in char_to_index]