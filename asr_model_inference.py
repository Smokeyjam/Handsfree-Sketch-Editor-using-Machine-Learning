# asr_inference.py (Improved with LSTM + Wav2Vec2 support)
import torch
import torch.nn as nn
import numpy as np
import os

from models.Speech_Recognition.lstm_asr.preprocess import extract_mfcc

# HuggingFace Wav2Vec2
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LSTM ASR SETUP ==========

# ✅ Paths to model and vocab
current_dir = os.path.dirname(os.path.abspath(__file__))
#data_dir = os.path.join(current_dir, "data")
MODEL_PATH = os.path.join(current_dir,"models\Speech_Recognition\lstm_asr\ctc_model.pth")
VOCAB_PATH = os.path.join(current_dir,"models\Speech_Recognition\lstm_asr\ctc_vocab.txt")
MAXLEN = 200

# ✅ Load vocab
with open(VOCAB_PATH, "r", encoding='utf-8') as f:
    vocab = list(f.read().strip())
index_to_char = {i: c for i, c in enumerate(vocab)}
blank_index = 0

# ✅ Define model (matching trained architecture)
class ASRModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(ASRModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm1(x)
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc_out(x)

# ✅ Load model
lstm_model = ASRModel(input_dim=13, vocab_size=len(vocab))
state_dict = torch.load(MODEL_PATH, map_location=device)
lstm_model.load_state_dict(state_dict)
lstm_model.to(device)
lstm_model.eval()

# ✅ Greedy CTC decoder
def ctc_decode(logits):
    pred_idx = logits.argmax(dim=-1).squeeze(0).tolist()
    prev = -1
    output = []
    for p in pred_idx:
        if p != prev and p != blank_index:
            output.append(index_to_char.get(p, ''))
        prev = p
    return ''.join(output)

# ✅ Transcribe using LSTM + MFCC model
def transcribe_audio(filepath):
    mfcc = extract_mfcc(filepath)
    mfcc = torch.tensor(mfcc, dtype=torch.float32)

    if mfcc.shape[0] > MAXLEN:
        mfcc = mfcc[:MAXLEN, :]
    else:
        mfcc = torch.nn.functional.pad(mfcc, (0, 0, 0, MAXLEN - mfcc.shape[0]))

    mfcc = mfcc.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = lstm_model(mfcc)
        transcription = ctc_decode(logits)

    if not transcription or len(transcription.strip()) < 3:
        return "[unrecognized speech]"

    return transcription.strip().lower()

# ========== Wav2Vec2 ASR SETUP ==========

W2V2_MODEL_NAME = "patrickvonplaten/wav2vec2-base-100h-with-lm"
LOCAL_W2V2_PATH = os.path.join(current_dir, "models", "Speech_Recognition", "wav2vec2-100h")

processor_w2v2 = Wav2Vec2Processor.from_pretrained(LOCAL_W2V2_PATH)
model_w2v2 = Wav2Vec2ForCTC.from_pretrained(LOCAL_W2V2_PATH).to(device)
model_w2v2.eval()

# ✅ Transcribe using Wav2Vec2
def transcribe_audio_wav2vec2(filepath):
    waveform, sample_rate = torchaudio.load(filepath)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    input_values = processor_w2v2(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(device)

    with torch.no_grad():
        logits = model_w2v2(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor_w2v2.decode(predicted_ids[0])

    if not transcription or len(transcription.strip()) < 3:
        return "[unrecognized speech]"

    return transcription.strip().lower()
