import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time

# Load model and processor (can switch to other pretrained models)
MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Parameters
SAMPLE_RATE = 16000  # Hz required by Wav2Vec2
DURATION = 5         # Seconds to record

def record_audio(duration, sample_rate):
    print(f"\n🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = audio.squeeze()
    print("✅ Recording complete.\n")
    return audio

def transcribe_audio(audio_np):
    # Preprocess audio
    inputs = processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    # Run inference
    with torch.no_grad():
        start_time = time.time()
        logits = model(input_values).logits
        inference_time = time.time() - start_time

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription, inference_time

def main():
    # Record live audio
    audio_np = record_audio(DURATION, SAMPLE_RATE)
    
    # Transcribe
    transcription, latency = transcribe_audio(audio_np)

    print("🗣️ Transcription:", transcription)
    print(f"⏱️ Inference Time: {latency:.2f} seconds")

if __name__ == "__main__":
    main()
