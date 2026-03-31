import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def calculate_metrics(pred, target):
    """
    pred, target: тензоры [Batch, 136] в диапазоне [0, 1]
    """
    # Переводим в [Batch, 68, 2]
    pred = pred.view(-1, 68, 2)
    target = target.view(-1, 68, 2)
    
    # Евклидово расстояние между точками
    dists = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1)) # [Batch, 68]
    
    # Средняя ошибка по всем точкам (Mean Error)
    mean_error = torch.mean(dists)
    
    # Расчет ошибки по зонам (используем маппинг из вашего датасета)
    # Индексы: глаза (36-47), губы (48-67)
    eye_error = torch.mean(dists[:, 36:48])
    lip_error = torch.mean(dists[:, 48:68])
    
    return mean_error.item(), eye_error.item(), lip_error.item()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_nme = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, landmarks in pbar:
        images, landmarks = images.to(device), landmarks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, landmarks)
        loss.backward()
        optimizer.step()
        
        nme, _, _ = calculate_metrics(outputs, landmarks)
        
        running_loss += loss.item()
        running_nme += nme
        
        pbar.set_postfix(loss=f"{loss.item():.5f}", nme=f"{nme:.4f}")
        
    return running_loss / len(loader), running_nme / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_nme = 0.0
    eye_nme = 0.0
    lip_nme = 0.0
    
    with torch.no_grad():
        for images, landmarks in loader:
            images, landmarks = images.to(device), landmarks.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, landmarks)
            nme, e_nme, l_nme = calculate_metrics(outputs, landmarks)
            
            running_loss += loss.item()
            running_nme += nme
            eye_nme += e_nme
            lip_nme += l_nme
            
    return (running_loss / len(loader), 
            running_nme / len(loader), 
            eye_nme / len(loader), 
            lip_nme / len(loader))

