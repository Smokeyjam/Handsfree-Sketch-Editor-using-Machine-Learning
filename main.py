# PyQt5 Drawing resources 
import sys, os, random, ndjson, csv
import torch
import time

from PyQt5.QtWidgets import QApplication, QPushButton, QComboBox, QLabel, QCheckBox
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from sketch_editor import SketchEditor
from drawing import Drawing, strokes_to_image
from stroke_item import StrokeItem

from sketch_model_inference import SketchClassifier
from cnn_quickdraw_classifier_resnet18 import QuickDrawClassifier

from Voice_Input.speech import SpeechRecognizer
from Voice_Input.record import record_audio
from asr_model_inference import transcribe_audio, transcribe_audio_wav2vec2

from command_parser import interpret_command  


# Globals for selector
voice_selector = None
voice_label = None
use_llm_checkbox = None  

subtitle_label = None
keyword_label = None


#have data folder with QuickDraw! sketches that the model was trained on
#Randomly select one of those drawings
def load_random_drawing(filepath):
    with open(filepath, 'r') as f:
        data = ndjson.load(f)
    drawing = random.choice(data)
    return drawing["drawing"]

def create_new_drawing(editor, data_dir, canvas_width=1280, canvas_height=720, scale=0.5):
    #random select a drawing
    filename = random.choice(os.listdir(data_dir))
    filepath = os.path.join(data_dir, filename)
    quickdraw_strokes = load_random_drawing(filepath)
    
    #prep file name
    label = filename.replace('.ndjson', '')

    #select random location to spawn
    max_offset_x = canvas_width - 300
    max_offset_y = canvas_height - 300
    offset_x = random.randint(50, max_offset_x)
    offset_y = random.randint(50, max_offset_y)

    print(f"Spawning '{label}' at ({offset_x}, {offset_y})")

    #intialise the drawing object, adding it the editor and list of drawings
    drawing = Drawing(label=label)
    drawing.build_from_quickdraw(quickdraw_strokes, offset_x=offset_x, offset_y=offset_y, scale=scale)
    drawing.add_to_scene(editor.scene)
    editor.drawings.append(drawing)

    return drawing

#a few editor commands 
def delete_selected(editor):
    if editor.selected_drawing is None:
        print("[WARN] No drawing selected!")
        return
    editor.selected_drawing.delete_from_scene(editor.scene)
    editor.drawings.remove(editor.selected_drawing)
    editor.selected_drawing = None
    print("[INFO] Selected drawing deleted.")

def move_selected(editor, dx, dy):
    if editor.selected_drawing is None:
        print("[WARN] No drawing selected to move!")
        return
    editor.selected_drawing.move_by(dx, dy)
    editor.viewport().update()
    print(f"[INFO] Moved drawing by ({dx}, {dy})")

def reclassify_selected(editor, classifier):
    if editor.selected_drawing is None:
        print("[WARN] No drawing selected!")
        return
    strokes = editor.selected_drawing.export_strokes()
    classifier.model.eval()
    with torch.no_grad():
        prediction = classifier.predict(strokes) #prediction = classifier.predict(strokes, strokes_to_image) for change to Xjay18's model
    print(f"[DEBUG] Predicted index: {prediction}")
    class_labels = [
        'candle', 'motorbike', 'cactus', 'crab', 'helicopter',
        'palm tree', 'fence', 'chair', 'toothbrush', 'giraffe'
    ]
    new_label = class_labels[prediction] if 0 <= prediction < len(class_labels) else f"Class ID: {prediction}"
    editor.selected_drawing.label = new_label
    if editor.selected_drawing.label_item:
        editor.selected_drawing.label_item.setText(new_label)
        editor.selected_drawing.update_label_position()
    print(f"[INFO] Drawing reclassified as '{new_label}'")


#Thread-safe VoiceWorker using QObject and signal (this is so latency can be counted)
class VoiceWorker(QObject):
   
    result_signal = pyqtSignal(str,str, list, object, float)


    def __init__(self, editor, data_dir, classifier):
        super().__init__()
        self.editor = editor
        self.data_dir = data_dir
        self.classifier = classifier
        self.recognizer = SpeechRecognizer()

    def run(self):
        while True:
            try:
                start_time = time.time()  #Track time before recording for end-to-end
                record_audio("input.wav", duration=5)
                selected_engine = voice_selector.currentText()
                if selected_engine == "OpenAI Whisper":
                    text = self.recognizer.transcribe("input.wav").lower()
                elif selected_engine == "Custom LSTM ASR":
                    text = transcribe_audio("input.wav").lower()
                elif selected_engine == "Light Wav2Vec2":
                    text = transcribe_audio_wav2vec2("input.wav").lower()
                else:
                    text = "[unknown engine]"

                import string
                text = text.strip().lower().translate(str.maketrans('', '', string.punctuation))
                if not text.strip():
                    print("[INFO] No speech detected — skipping command parsing.")
                    continue

                select_label = None
                if use_llm_checkbox and use_llm_checkbox.isChecked():
                    commands = interpret_command(text)
                    print(f"[LLM] Commands detected: {commands}")
                else:
                    commands = []
                    if "new drawing" in text or "spawn" in text:
                        commands.append("spawn")
                    if "reclassify" in text or "rename" in text:
                        commands.append("reclassify")
                    if "delete" in text or "remove" in text:
                        commands.append("delete")
                    if "move up" in text:
                        commands.append("move_up")
                    if "move down" in text:
                        commands.append("move_down")
                    if "move left" in text:
                        commands.append("move_left")
                    if "move right" in text:
                        commands.append("move_right")
                    if text.startswith("select "):
                        select_label = text.replace("select ", "").strip()
                        commands.append("select")

                #everything to main thread for handling
                self.result_signal.emit(text, selected_engine, commands, select_label, start_time)

            except Exception as e:
                print(f"[ERROR] Voice thread error: {e}")
            time.sleep(1)

# Slot function extra for log latency (I was running errors with running time on unsafe threads)
def handle_voice_result(text, engine, commands, select_label, start_time):
    subtitle_label.setText(text)
    voice_label.setText(f"Current ASR: {engine}")
    keyword_label.setText(f"Detected commands: {', '.join(commands)}")

    if "select" in commands and select_label:
        editor.select_drawing_by_name(select_label)

    for command in commands:
        if command == "spawn":
            create_new_drawing(editor, data_dir)
        elif command == "reclassify":
            reclassify_selected(editor, classifier)
        elif command == "delete":
            delete_selected(editor)
        elif command == "move_up":
            move_selected(editor, 0, -100)
        elif command == "move_down":
            move_selected(editor, 0, 100)
        elif command == "move_left":
            move_selected(editor, -100, 0)
        elif command == "move_right":
            move_selected(editor, 100, 0)

    #Compute and log latency
    end_time = time.time()
    latency = end_time - start_time
    commands_str = ', '.join(commands)

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_log.csv")
    log_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not log_exists:
            writer.writerow(["ASR Engine", "Transcribed Text", "Commands", "Latency (s)"])
        writer.writerow([engine, text, commands_str, round(latency, 2)])

    print(f"[METRIC] Total latency from audio start to command execution: {latency:.2f} seconds")
    print(f"[LOG] Logged to latency_log.csv")


#main where all the objects and buttons are intialised
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")

    app = QApplication(sys.argv)
    editor = SketchEditor()
    editor.setWindowTitle("Sketch Editor")
    editor.resize(1280, 720)

    Select_Button = QPushButton(editor)
    Select_Button.setText("Selection/Draw")
    Select_Button.move(10, 10)
    Select_Button.resize(150, 40)
    Select_Button.clicked.connect(editor.toggle_mode)

    New_Drawing_Button = QPushButton(editor)
    New_Drawing_Button.setText("New Drawing")
    New_Drawing_Button.move(10, 110)
    New_Drawing_Button.resize(150, 40)
    New_Drawing_Button.clicked.connect(lambda: create_new_drawing(editor, data_dir))

    Delete_Button = QPushButton(editor)
    Delete_Button.setText("Delete Drawing")
    Delete_Button.move(10, 160)
    Delete_Button.resize(150, 40)
    Delete_Button.clicked.connect(lambda: delete_selected(editor))

    classifier = SketchClassifier(
        model_dir=os.path.join(current_dir, "models/Sketch_Recognition/finetuned_quickdraw_model"),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    #classifier = QuickDrawClassifier(
    #    model_path=os.path.join(current_dir, "models/Sketch_Recognition/quickdraw_resnet18/quickdraw_resnet18.pytorch"),
    #    device='cuda' if torch.cuda.is_available() else 'cpu'
    #)

    Reclassify_Button = QPushButton(editor)
    Reclassify_Button.setText("Reclassify Drawing")
    Reclassify_Button.move(10, 210)
    Reclassify_Button.resize(150, 40)
    Reclassify_Button.clicked.connect(lambda: reclassify_selected(editor, classifier))

    voice_selector = QComboBox(editor)
    voice_selector.addItem("OpenAI Whisper")
    voice_selector.addItem("Custom LSTM ASR")
    voice_selector.addItem("Light Wav2Vec2")
    voice_selector.move(10, 310)
    voice_selector.resize(200, 40)

    voice_label = QLabel("Current ASR: OpenAI Whisper", editor)
    voice_label.move(10, 360)
    voice_label.resize(200, 30)

    use_llm_checkbox = QCheckBox("Use LLM Command Parser", editor)
    use_llm_checkbox.setChecked(False)
    use_llm_checkbox.move(10, 400)
    use_llm_checkbox.resize(250, 30)

    subtitle_label = QLabel("", editor)
    subtitle_label.move(300, 640)
    subtitle_label.resize(900, 40)
    subtitle_label.setStyleSheet("font-size: 18px; color: white; background-color: rgba(0,0,0,0.6); padding: 5px;")

    keyword_label = QLabel("", editor)
    keyword_label.move(300, 680)
    keyword_label.resize(900, 30)
    keyword_label.setStyleSheet("font-size: 16px; color: yellow; background-color: rgba(0,0,0,0.5); padding: 3px;")

    editor.show()
    create_new_drawing(editor, data_dir)

    #voice recognition using QThread
    voice_worker = VoiceWorker(editor, data_dir, classifier)
    voice_thread = QThread()
    voice_worker.moveToThread(voice_thread)
    voice_worker.result_signal.connect(handle_voice_result)
    voice_thread.started.connect(voice_worker.run)
    voice_thread.start()

    sys.exit(app.exec_())
