# train_ctc_model.py (Improved BiLSTM CTC training with vocab alignment)
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from preprocess import extract_mfcc
from ctc_model import CTCModel
from utils import text_to_labels, get_vocab
from tqdm import tqdm, trange
import torch.optim as optim
import jiwer

# CONFIG
LIBRISPEECH_DIR = r"C:/Users/loone/Desktop/Uni_work/Year 3/Project/Assessment 3/Artifact Creation/Artifact_V7/lstm_asr_Training/LibriSpeech/train-clean-100"
MAX_SAMPLES = 10000
MAXLEN = 500

X, y_text = [], []

print("[INFO] Scanning dataset...")
all_flacs = []
transcripts = {}
for root, dirs, files in os.walk(LIBRISPEECH_DIR):
    for file in files:
        if file.endswith(".trans.txt"):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        key, sentence = parts
                        transcripts[key] = sentence.upper()
        elif file.endswith(".flac"):
            all_flacs.append((root, file))

all_flacs = all_flacs[:MAX_SAMPLES]

print("[INFO] Extracting MFCC features and labels...")
for root, file in tqdm(all_flacs, desc="Processing"):
    file_id = file.replace(".flac", "")
    if file_id not in transcripts:
        continue
    mfcc = extract_mfcc(os.path.join(root, file))
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)  # Normalize
    X.append(torch.tensor(mfcc, dtype=torch.float32))
    y_text.append(transcripts[file_id])

# ✅ Character labels: ensure underscore _ is at index 0
vocab, _ = get_vocab(y_text)
vocab = sorted(set(vocab) - set("'"))  # Remove apostrophe if not needed
vocab = ["_"] + vocab  # Blank index first
char_to_index = {c: i for i, c in enumerate(vocab)}
index_to_char = {i: c for i, c in enumerate(vocab)}
blank_index = 0

# ✅ Encode labels
y = [torch.tensor(text_to_labels(text, char_to_index), dtype=torch.long) for text in y_text]

# ✅ Pad MFCCs
X = [x[:MAXLEN] if x.shape[0] > MAXLEN else torch.nn.functional.pad(x, (0, 0, 0, MAXLEN - x.shape[0])) for x in X]
X = torch.stack(X)

# ✅ Pad labels
y_lengths = torch.tensor([len(seq) for seq in y], dtype=torch.long)
y = pad_sequence(y, batch_first=True, padding_value=0)

# Input lengths
x_lengths = torch.full(size=(len(X),), fill_value=MAXLEN, dtype=torch.long)

# ✅ Train/val split
X_train, X_val, y_train, y_val, xlen_train, xlen_val, ylen_train, ylen_val = train_test_split(
    X, y, x_lengths, y_lengths, test_size=0.2, random_state=42
)

# DataLoaders
train_dataset = TensorDataset(X_train, y_train, xlen_train, ylen_train)
val_dataset = TensorDataset(X_val, y_val, xlen_val, ylen_val)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

print("[INFO] Building PyTorch CTC model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTCModel(input_dim=13, vocab_size=len(vocab)).to(device)
criterion = nn.CTCLoss(blank=blank_index, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

print("[INFO] Training...")
os.makedirs("lstm_asr", exist_ok=True)
log_file = open("lstm_asr/ctc_training_log.csv", "w")
log_file.write("epoch,train_loss,val_loss,wer\n")

def decode_prediction(preds, blank_index):
    decoded = []
    for seq in preds:
        collapsed = torch.unique_consecutive(seq)
        filtered = [index_to_char[i.item()] for i in collapsed if i.item() != blank_index]
        decoded.append(''.join(filtered))
    return decoded

for epoch in trange(10, desc="Epoch"):
    model.train()
    train_loss = 0.0
    for inputs, targets, input_lengths, target_lengths in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

        outputs = model(inputs)  # [B, T, C]
        outputs = outputs.permute(1, 0, 2)  # [T, B, C]

        optimizer.zero_grad()
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    hypotheses, references = [], []
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)  # [B, T]
            loss = criterion(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            val_loss += loss.item()

            decoded_preds = decode_prediction(preds, blank_index=blank_index)
            for i, tgt_len in enumerate(target_lengths):
                true_str = ''.join([index_to_char.get(t.item(), '') for t in targets[i][:tgt_len] if t.item() != 0])
                pred_str = decoded_preds[i]
                if true_str.strip() and any(c.isalnum() for c in true_str):
                    references.append(true_str)
                    hypotheses.append(pred_str)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    wer_score = jiwer.wer(references, hypotheses)
    print(f"Epoch {epoch+1}/10 - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, WER: {wer_score:.4f}")
    log_file.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{wer_score:.4f}\n")
    scheduler.step(avg_val_loss)

log_file.close()

print("[INFO] Saving model and vocabulary...")
torch.save(model.state_dict(), "lstm_asr/ctc_model.pth")
with open("lstm_asr/ctc_vocab.txt", "w", encoding='utf-8') as f:
    f.write("".join(vocab))
