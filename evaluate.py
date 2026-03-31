import os
import cv2
import dlib
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from models import FaceLandmarkModel

# 1. Настройки и загрузка
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PADDING = 0.2
IMG_SIZE = 224

# Загружаем твою модель
model = FaceLandmarkModel().to(DEVICE)
model.load_state_dict(torch.load("mobileNet_models/best_model.pth", map_location=DEVICE))
model.eval()

# Загружаем инструменты DLIB
detector = dlib.get_frontal_face_detector()
# Скачай этот файл: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_nme(pred_pts, gt_pts, rect):
    """Расчет ошибки по формуле из ТЗ"""
    w, h = rect.width(), rect.height()
    normalization = np.sqrt(w * h)
    
    dists = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return np.mean(dists) / normalization

def evaluate_dataset(root_dir, is_menpo=False):
    my_model_errors = []
    dlib_errors = []
    
    # Ищем все .pts файлы
    pts_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if f.endswith('.pts')]
    
    for pts_path in tqdm(pts_files, desc=f"Evaluating {os.path.basename(root_dir)}"):
        img_path = pts_path.replace('.pts', '.jpg')
        if not os.path.exists(img_path): img_path = pts_path.replace('.pts', '.png')
        
        # Читаем картинку и GT точки
        image = cv2.imread(img_path)
        if image is None: continue
        h_orig, w_orig = image.shape[:2]
        
        with open(pts_path, 'r') as f:
            lines = f.readlines()
            gt_pts = np.array([line.split() for line in lines if len(line.split()) == 2 and line.split()[0][0].isdigit()], dtype=np.float32)
        
        if len(gt_pts) != 68: continue # Пропускаем профили Menpo

        # --- Детекция лица через DLIB ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) == 0: continue # Пропускаем, если dlib не нашел лицо
        
        # Берем самое подходящее лицо
        gt_center = gt_pts.mean(axis=0)
        rect = min(rects, key=lambda r: np.linalg.norm(gt_center - np.array([(r.left()+r.right())/2, (r.top()+r.bottom())/2])))
        
        # --- 1. ТВОЯ МОДЕЛЬ ---
        # Делаем кроп с тем же падингом, что при обучении
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        w_box, h_box = x2 - x1, y2 - y1
        x1_p = int(max(0, x1 - w_box * PADDING))
        y1_p = int(max(0, y1 - h_box * PADDING))
        x2_p = int(min(w_orig, x2 + w_box * PADDING))
        y2_p = int(min(h_orig, y2 + h_box * PADDING))
        
        crop = image[y1_p:y2_p, x1_p:x2_p]
        crop_h, crop_w = crop.shape[:2]
        
        # Подготовка тензора
        input_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (IMG_SIZE, IMG_SIZE))
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float().div(255.0)
        # Нормализация ImageNet
        input_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input_tensor)
        
        with torch.no_grad():
            pred = model(input_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy().reshape(68, 2)
        
        # ПЕРЕВОД В ОРИГИНАЛЬНЫЕ КООРДИНАТЫ
        pred_orig = pred.copy()
        pred_orig[:, 0] = pred_orig[:, 0] * crop_w + x1_p
        pred_orig[:, 1] = pred_orig[:, 1] * crop_h + y1_p
        
        my_model_errors.append(get_nme(pred_orig, gt_pts, rect))
        
        # --- 2. DLIB BASELINE (только для Menpo) ---
        if is_menpo:
            shape = dlib_predictor(gray, rect)
            dlib_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
            dlib_errors.append(get_nme(dlib_pts, gt_pts, rect))
            
    return my_model_errors, dlib_errors

def plot_ced(errors_dict, title, save_name):
    plt.figure(figsize=(8, 6))
    
    for label, errors in errors_dict.items():
        if not errors: continue
        errors = np.sort(errors)
        # Считаем процент изображений с ошибкой меньше X
        x = np.linspace(0, 0.08, 100)
        y = [np.mean(errors <= threshold) for threshold in x]
        
        # Считаем AUC
        auc = np.trapezoid(y, x) / 0.08
        plt.plot(x, y, label=f"{label} (AUC: {auc:.4f})")
        
    plt.xlim(0, 0.08)
    plt.ylim(0, 1.0)
    plt.xlabel("Normalized Error (NME)")
    plt.ylabel("Fraction of Images")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)
    plt.show()

# --- ЗАПУСК ОЦЕНКИ ---
# 300W Test
errors_300w, _ = evaluate_dataset("300W/test")
plot_ced({"My MobileNetV2": errors_300w}, "CED Curve - 300W Test", "ced_300w.png")

# Menpo Test
errors_menpo, dlib_baseline = evaluate_dataset("Menpo/test", is_menpo=True)
plot_ced({
    "My MobileNetV2": errors_menpo,
    "Dlib Baseline": dlib_baseline
}, "CED Curve - Menpo Test", "ced_menpo.png")