# plot_ctc_metrics.py
import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("C:/Users/loone/Desktop/Uni_work/Year 3/Project/Assessment 3/Artifact Creation/Artifact_V7/lstm_asr_Training/lstm_asr/ctc_training_log.csv")

plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(log['epoch'], log['train_loss'], label='Train Loss')
plt.plot(log['epoch'], log['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CTC Model Loss')
plt.legend()
plt.grid(True)

# Plot WER
plt.subplot(1, 2, 2)
plt.plot(log['epoch'], log['wer'], label='WER', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Word Error Rate')
plt.title('CTC Model WER')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
