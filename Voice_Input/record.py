#records 5 seconds of audio from the microphone
import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="input.wav", duration=5, fs=44100):
    print("[INFO] Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"[INFO] Saved recording to {filename}")
