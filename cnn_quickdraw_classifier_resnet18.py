import torch
import torch.nn as nn
import numpy as np

class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(QuickDrawCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),  # fc.0
            nn.Linear(512, num_classes)  # fc.1
        )

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.layer3(x)
        x = nn.functional.adaptive_avg_pool2d(x, (3, 3))  # ensures output is 3x3
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class QuickDrawClassifier:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = QuickDrawCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, strokes, transform_func):
        image = transform_func(strokes)  # shape (28, 28)
        tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # shape [1, 1, 28, 28]
        with torch.no_grad():
            output = self.model(tensor)
            return output.argmax(dim=1).item()
