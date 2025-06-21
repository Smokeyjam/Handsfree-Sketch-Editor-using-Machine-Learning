# speech.py loads whisper model into memory and sends audio file for recognition
import whisper

class SpeechRecognizer:
    def __init__(self, model_size="base"):
        print(f"[INFO] Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path):
        print(f"[INFO] Transcribing {audio_path}")
        result = self.model.transcribe(audio_path)
        return result["text"]
