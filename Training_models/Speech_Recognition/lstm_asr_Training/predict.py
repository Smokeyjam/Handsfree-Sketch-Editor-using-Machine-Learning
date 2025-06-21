# predict.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from preprocess import extract_mfcc
from utils import get_vocab
import sys

# Load model and vocab
model = load_model("lstm_asr/ctc_model.h5", compile=False)
with open("lstm_asr/ctc_vocab.txt", "r", encoding='utf-8') as f:
    vocab = list(f.read())
index_to_char = {i: c for i, c in enumerate(vocab)}

# Decode greedy prediction
def ctc_greedy_decode(pred):
    pred_idx = np.argmax(pred, axis=-1)[0]
    prev = -1
    output = []
    for p in pred_idx:
        if p != prev and p < len(vocab):
            output.append(index_to_char.get(p, ''))
        prev = p
    return ''.join(output)

# Predict from audio
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/audio.flac")
        sys.exit(1)

    path = sys.argv[1]
    mfcc = extract_mfcc(path)
    mfcc = pad_sequences([mfcc], padding="post", maxlen=200, dtype='float32')
    mfcc = np.expand_dims(mfcc, -1)  # [1, time, features, 1]

    pred = model.predict(mfcc)
    transcription = ctc_greedy_decode(pred)
    print("[TRANSCRIPTION]", transcription)
