# lstm_asr/infer.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from preprocess import extract_mfcc

def load_labels(path="lstm_asr/labels.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def predict_audio(file_path, model_path="lstm_asr/lstm_model.h5"):
    mfcc = extract_mfcc(file_path)
    X = pad_sequences([mfcc], maxlen=200, padding="post", dtype='float32')
    model = load_model(model_path)
    preds = model.predict(X)
    return np.argmax(preds), preds

if __name__ == "__main__":
    audio_file = "input.wav"  # Replace with your recording
    pred_idx, pred_probs = predict_audio(audio_file)
    labels = load_labels()
    print(f"[PREDICTION] Word: {labels[pred_idx]} ({pred_probs[0][pred_idx]:.2f})")
