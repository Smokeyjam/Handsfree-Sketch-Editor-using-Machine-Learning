# lstm_asr/preprocess.py
import librosa
import numpy as np
import soundfile as sf

def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    audio, file_sr = sf.read(file_path)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # [Time, Features]
