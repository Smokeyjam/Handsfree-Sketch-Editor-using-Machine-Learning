import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForImageClassification
import ndjson
from tqdm import tqdm
from drawing import strokes_to_image  # Assumes this is in your project
import csv

# --- Config ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
output_dir = os.path.join(current_dir, "finetuned_quickdraw_model")
log_path = os.path.join(output_dir, "training_log.csv")
class_labels = ['candle', 'motorbike', 'cactus', 'crab', 'helicopter',
                'palm tree', 'fence', 'chair', 'toothbrush', 'giraffe']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
learning_rate = 3e-4
num_epochs = 20
max_samples_per_class = 1000

# --- Data Loading ---
X, y = [], []
print("Loading and preprocessing sketch data...")

for class_id, class_name in enumerate(class_labels):
    file_path = os.path.join(data_dir, f"{class_name}.ndjson")
    with open(file_path, 'r') as f:
        sketches = ndjson.load(f)

    for sketch in sketches[:max_samples_per_class]:
        strokes = sketch["drawing"]
        image = strokes_to_image(strokes, canvas_size=224)
        tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        X.append(tensor)
        y.append(class_id)

X = torch.stack(X)
y = torch.tensor(y)

# --- Dataset Split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# --- Model Setup ---
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(class_labels)
).to(device)

# Freeze all but classifier
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

# --- Training Setup ---
criterion = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# --- Logging Setup ---
os.makedirs(output_dir, exist_ok=True)
with open(log_path, "w", newline='') as log_f:
    writer = csv.writer(log_f)
    writer.writerow(["epoch", "train_loss", "train_mse", "val_accuracy", "val_mse"])

# --- Training Loop ---
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_mse = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)

            # MSE calculation
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=len(class_labels)).float()
            mse = mse_loss(probs, labels_one_hot.to(device))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += mse.item()

    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Train MSE = {avg_mse:.4f}")

    # --- Validation ---
    model.eval()
    correct, total, val_mse = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=images)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=len(class_labels)).float()
                val_mse += mse_loss(probs, labels_one_hot.to(device)).item()

    val_acc = 100 * correct / total
    avg_val_mse = val_mse / len(val_loader)
    print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.2f}%, Val MSE = {avg_val_mse:.4f}")

    with open(log_path, "a", newline='') as log_f:
        writer = csv.writer(log_f)
        writer.writerow([epoch+1, f"{avg_loss:.4f}", f"{avg_mse:.4f}", f"{val_acc:.2f}", f"{avg_val_mse:.4f}"])

# --- Save Model ---
torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
model.config.id2label = {i: label for i, label in enumerate(class_labels)}
model.config.label2id = {label: i for i, label in enumerate(class_labels)}
model.config.save_pretrained(output_dir)
print("✅ Model and config saved.")
