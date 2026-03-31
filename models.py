import torch
import torch.nn as nn
from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class FaceLandmarkModel(nn.Module):
    def __init__(self):
        super(FaceLandmarkModel, self).__init__()
        # Загружаем предобученный MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # У MobileNetV2 последний слой находится в self.model.classifier[1]
        # Извлекаем количество входных признаков (там 1280)
        in_features = self.model.classifier[1].in_features
        
        # Заменяем на новый слой: 136 выходных нейронов
        self.model.classifier[1] = nn.Linear(in_features, 136)

    def forward(self, x):
        return self.model(x)


class ResNetFaceLandmark(nn.Module):
    def __init__(self):
        super(ResNetFaceLandmark, self).__init__()
        # Загружаем предобученный ResNet-18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # У ResNet последний слой называется 'fc'
        in_features = self.model.fc.in_features
        
        # Меняем его на 136 выходов
        self.model.fc = nn.Linear(in_features, 136)

    def forward(self, x):
        return self.model(x)

# Проверка:
model = FaceLandmarkModel()
print(f"Вес модели: {sum(p.numel() for p in model.parameters()) * 4 / (1024**2):.2f} MB")

model = ResNetFaceLandmark()
print(f"Вес модели: {sum(p.numel() for p in model.parameters()) * 4 / (1024**2):.2f} MB")
# Должно быть ~9-14 MB