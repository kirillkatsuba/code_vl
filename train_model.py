# import torch
# import torch.nn as nn
# import torch.optim as optim
# from dataset_loader import get_dataloaders, get_concat_dataloaders
# from training import calculate_metrics, train_one_epoch, validate
# from models import FaceLandmarkModel, ResNetFaceLandmark

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else ( "mps" if torch.backends.mps.is_available else "cpu"))


# #=====================================
# model = FaceLandmarkModel().to(DEVICE)
# #=====================================


# print(DEVICE)
# criterion = nn.SmoothL1Loss() # Устойчива к выбросам
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)

# train_loader, val_loader = get_dataloaders('Menpo/train', 'Menpo/test', batch_size=32)
# train_loader, val_loader = get_concat_dataloaders(['Menpo/train', '300W/train'], 'Menpo/train', batch_size=32)

# NUM_EPOCHS = 40
# best_val_nme = float('inf')

# # История для графиков
# history = {'train_loss': [], 'val_loss': [], 'val_nme': []}

# for epoch in range(NUM_EPOCHS):
#     print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
#     train_loss, train_nme = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
#     val_loss, val_nme, val_eye, val_lip = validate(model, val_loader, criterion, DEVICE)
    
#     scheduler.step(val_loss)
    
#     # Сохраняем историю
#     history['train_loss'].append(train_loss)
#     history['val_loss'].append(val_loss)
#     history['val_nme'].append(val_nme)
    
#     print(f"Summary: Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
#     print(f"Metrics: Total NME: {val_nme:.4f} | Eyes NME: {val_eye:.4f} | Lips NME: {val_lip:.4f}")
    
#     if val_nme < best_val_nme:
#         best_val_nme = val_nme
#         torch.save(model.state_dict(), "mobileNet_models_menpo/best_model.pth")
#         print("✅ Модель сохранена!")
#     torch.save(model.state_dict(), f"mobileNet_models_menpo/model_{epoch}.pth")
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_dataloaders, get_concat_dataloaders
from training import calculate_metrics, train_one_epoch, validate
from models import FaceLandmarkModel, ResNetFaceLandmark

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['mobilenet', 'resnet'], default='mobilenet')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_data', type=str, choices=['menpo', '300w', 'both'], default='both')
parser.add_argument('--val_data', type=str, choices=['menpo', '300w'], default='menpo')
args = parser.parse_args()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)

if args.model == 'resnet':
    model = ResNetFaceLandmark().to(DEVICE)
else:
    model = FaceLandmarkModel().to(DEVICE)

print(DEVICE)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)

if args.val_data == 'menpo':
    val_path = '/kaggle/input/datasets/kirillkatsuba/traindataset/Menpo/test'
else:
    val_path = '/kaggle/input/datasets/kirillkatsuba/traindataset/300W/test'

if args.train_data == 'menpo':
    train_loader, val_loader = get_dataloaders('Menpo/train', val_path, batch_size=args.batch_size)
elif args.train_data == '300w':
    train_loader, val_loader = get_dataloaders('300W/train', val_path, batch_size=args.batch_size)
else:
    train_loader, val_loader = get_concat_dataloaders(['Menpo/train', '300W/train'], val_path, batch_size=args.batch_size)

NUM_EPOCHS = 40
best_val_nme = float('inf')

history = {'train_loss': [], 'val_loss': [], 'val_nme': []}

save_path = f"{args.model}_models_{args.train_data}"
os.makedirs(save_path, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    train_loss, train_nme = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_nme, val_eye, val_lip = validate(model, val_loader, criterion, DEVICE)
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_nme'].append(val_nme)
    
    print(f"Summary: Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
    print(f"Metrics: Total NME: {val_nme:.4f} | Eyes NME: {val_eye:.4f} | Lips NME: {val_lip:.4f}")
    
    if val_nme < best_val_nme:
        best_val_nme = val_nme
        torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        print("✅ Модель сохранена!")
    torch.save(model.state_dict(), f"{save_path}/model_{epoch}.pth")