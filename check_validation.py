import os
import cv2
import dlib
import torch
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from models import FaceLandmarkModel

# 1. Загрузка модели
DEVICE = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceLandmarkModel() # Используем вашу архитектуру MobileNetV2
model.load_state_dict(torch.load("mobileNet_models/best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 2. Инициализация dlib и трансформаций
detector = dlib.get_frontal_face_detector()
img_size = 224
padding = 0.2 # Тот же паддинг, что был при обучении!

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_random_samples(folder_path, num_samples=4):
    # Ищем все картинки
    img_files = glob.glob(os.path.join(folder_path, "**/*.jpg"), recursive=True) + \
                glob.glob(os.path.join(folder_path, "**/*.png"), recursive=True)
    
    samples = random.sample(img_files, num_samples)
    
    plt.figure(figsize=(18, 5 * ((num_samples+3)//4)))
    
    for i, img_path in enumerate(samples):
        # А) Загрузка изображения и точек (GT)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pts_path = img_path.rsplit('.', 1)[0] + '.pts'
        gt_landmarks = None
        if os.path.exists(pts_path):
            with open(pts_path, 'r') as f:
                lines = f.readlines()
                gt_landmarks = np.array([line.split() for line in lines if len(line.split()) == 2 and line.split()[0][0].isdigit()], dtype=np.float32)

        # Б) Детекция лица через DLIB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        
        if len(rects) > 0:
            # Выбираем самое большое лицо или то, что ближе к центру (если есть GT)
            rect = rects[0]
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        elif gt_landmarks is not None:
            # Если dlib не нашел, используем рамку по точкам (fallback)
            x1, y1 = gt_landmarks.min(axis=0).astype(int)
            x2, y2 = gt_landmarks.max(axis=0).astype(int)
        else:
            continue # Пропускаем, если лица нет совсем

        # В) Кроп с паддингом (точно так же, как в Dataset)
        h_orig, w_orig = image.shape[:2]
        w_box, h_box = x2 - x1, y2 - y1
        x1_p = int(max(0, x1 - w_box * padding))
        y1_p = int(max(0, y1 - h_box * padding))
        x2_p = int(min(w_orig, x2 + w_box * padding))
        y2_p = int(min(h_orig, y2 + h_box * padding))
        
        face_crop = image_rgb[y1_p:y2_p, x1_p:x2_p]
        crop_h, crop_w = face_crop.shape[:2]

        # Г) Подготовка для модели
        input_tensor = img_transform(face_crop).unsqueeze(0).to(DEVICE)
        
        # Д) Инференс
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy().reshape(68, 2)
        
        # Пересчет предсказаний в пиксели кропа (0..1 -> 0..224)
        pred_px = prediction * img_size 
        
        # Е) Если есть GT, пересчитываем их тоже в систему координат кропа
        if gt_landmarks is not None:
            gt_px = gt_landmarks.copy()
            gt_px[:, 0] = (gt_px[:, 0] - x1_p) / crop_w * img_size
            gt_px[:, 1] = (gt_px[:, 1] - y1_p) / crop_h * img_size

        # Ж) Визуализация кропа 224x224
        # Делаем ресайз самого изображения для отрисовки точек поверх
        disp_crop = cv2.resize(face_crop, (img_size, img_size))
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(disp_crop)
        
        # Рисуем предсказания (Красные)
        plt.scatter(pred_px[:, 0], pred_px[:, 1], s=15, c='red', marker='o', label='Pred')
        
        # Рисуем Ground Truth (Зеленые), если они есть
        if gt_landmarks is not None:
            plt.scatter(gt_px[:, 0], gt_px[:, 1], s=10, c='lime', marker='.', label='GT', alpha=0.6)
            
        plt.title(f"300W Test: {os.path.basename(img_path)}")
        plt.axis('off')
        if i == 0: plt.legend()

    plt.tight_layout()
    plt.show()

# Запускаем!
predict_random_samples('300W/test', num_samples=4)