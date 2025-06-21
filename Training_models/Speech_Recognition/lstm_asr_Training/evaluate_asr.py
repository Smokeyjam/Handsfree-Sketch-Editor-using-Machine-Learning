# evaluate_asr.py (PyTorch version)
import os
import torch
import numpy as np
from jiwer import wer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from preprocess import extract_mfcc
from utils import get_vocab
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pad_sequence

from ctc_model import CTCModel as ASRModel # ✅ Make sure this matches your actual model script


# CONFIG
TEST_DIR = r"C:\Users\loone\Desktop\Uni_work\Year 3\Project\Assessment 3\Artifact Creation\Artifact_V7\lstm_asr\LibriSpeech\train-clean-100"
MODEL_PATH = r"C:\Users\loone\Desktop\Uni_work\Year 3\Project\Assessment 3\Artifact Creation\Artifact_V7\lstm_asr\lstm_asr\ctc_model.pth"
VOCAB_PATH = r"C:\Users\loone\Desktop\Uni_work\Year 3\Project\Assessment 3\Artifact Creation\Artifact_V7\lstm_asr\lstm_asr\ctc_vocab.txt"
MAX_TEST_SAMPLES = 500
MAXLEN = 200

print("[INFO] Loading model and vocab...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(VOCAB_PATH, "r", encoding='utf-8') as f:
    vocab = list(f.read())
vocab_size = len(vocab)
index_to_char = {i: c for i, c in enumerate(vocab)}

model = ASRModel(input_dim=13, vocab_size=vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# Decode greedy prediction
def ctc_greedy_decode(logits):
    pred_idx = logits.argmax(dim=-1).squeeze(0).tolist()
    prev = -1
    output = []
    for p in pred_idx:
        if p != prev and p < len(vocab):
            output.append(index_to_char.get(p, ''))
        prev = p
    return ''.join(output)

# Load test data
print("[INFO] Loading test data...")
all_flacs, transcripts = [], {}
for root, dirs, files in os.walk(TEST_DIR):
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

all_flacs = all_flacs[:MAX_TEST_SAMPLES]

# Inference
y_true, y_pred = [], []

print("[INFO] Running inference on test set...")
for root, file in tqdm(all_flacs):
    file_id = file.replace(".flac", "")
    if file_id not in transcripts:
        continue

    mfcc = extract_mfcc(os.path.join(root, file))
    mfcc = torch.tensor(mfcc, dtype=torch.float32)
    if mfcc.shape[0] > MAXLEN:
        mfcc = mfcc[:MAXLEN, :]
    else:
        pad_len = MAXLEN - mfcc.shape[0]
        mfcc = torch.nn.functional.pad(mfcc, (0, 0, 0, pad_len))

    mfcc = mfcc.unsqueeze(0).to(device)  # [1, T, F]

    with torch.no_grad():
        logits = model(mfcc)  # [1, T, C]
        transcription = ctc_greedy_decode(logits.cpu())

    y_pred.append(transcription.strip())
    y_true.append(transcripts[file_id].strip())

# Compute WER
overall_wer = wer(y_true, y_pred)
print(f"[RESULT] WER: {overall_wer:.4f}")

# Save predictions
os.makedirs("lstm_asr", exist_ok=True)
with open("lstm_asr/eval_predictions.txt", "w", encoding='utf-8') as f:
    for ref, hyp in zip(y_true, y_pred):
        f.write(f"REF: {ref}\nHYP: {hyp}\n\n")

# Confusion Matrix (character level)
true_chars = list(''.join(y_true))
pred_chars = list(''.join(y_pred))
if len(true_chars) != len(pred_chars):
    print(f"[WARN] Skipping confusion matrix: unequal lengths ({len(true_chars)} vs {len(pred_chars)})")
else:
    labels = sorted(set(true_chars + pred_chars))
    cm = confusion_matrix(true_chars, pred_chars, labels=labels)

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=False, xticks_rotation='vertical', cmap='Blues')
    plt.title("Character-Level Confusion Matrix")
    plt.tight_layout()
    plt.savefig("lstm_asr/confusion_matrix.png")
    plt.show()

