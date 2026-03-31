import os
import cv2
import dlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt

import os
import cv2
import dlib
import numpy as np
import torch
import json
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class FaceLandmarksDataset(Dataset):
    FLIP_INDICES = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (36, 45), (37, 44), (38, 43), (39, 42), (41, 46), (40, 47),
        (31, 35), (32, 34),
        (48, 54), (49, 53), (50, 52), (59, 55), (58, 56),
        (60, 64), (61, 63), (67, 65)
    ]

    def __init__(self, root_dir, img_size=224, padding=0.2, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.padding = padding
        self.is_train = is_train
        self.detector = dlib.get_frontal_face_detector()
        self.preprocessed_data = []
        
        self._prepare_data()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _read_pts(self, pts_path):
        with open(pts_path, 'r') as f:
            lines = f.readlines()
            landmarks = [line.split() for line in lines if len(line.split()) == 2 and line.split()[0][0].isdigit()]
        return np.array(landmarks, dtype=np.float32)

    def _prepare_data(self):
        # cache_path = os.path.join(self.root_dir, "face_data_cache.json")
        
        # # 1. Попытка загрузки из кэша
        # if os.path.exists(cache_path):
        #     print(f"--- Загрузка кэша из {cache_path} ---")
        #     with open(cache_path, 'r') as f:
        #         cached_list = json.load(f)
        #         for item in cached_list:
        #             # Конвертируем обратно в numpy для удобства работы в __getitem__
        #             item['landmarks'] = np.array(item['landmarks'], dtype=np.float32)
        #             self.preprocessed_data.append(item)
        #     print(f"Успешно загружено из кэша: {len(self.preprocessed_data)} образцов.")
        #     return

        # 2. Если кэша нет — выполняем детекцию
        search_pattern = os.path.join(self.root_dir, "**/*.pts")
        all_pts_files = glob.glob(search_pattern, recursive=True)
        
        print(f"--- Кэш не найден. Начало детекции лиц в: {self.root_dir} ---")
        
        for pts_path in tqdm(all_pts_files, desc="Препроцессинг"):
            img_path = pts_path.replace('.pts', '.jpg')
            if not os.path.exists(img_path):
                img_path = pts_path.replace('.pts', '.png')
            
            if os.path.exists(img_path):
                landmarks = self._read_pts(pts_path)
                if len(landmarks) != 68:
                    continue
                
                image = cv2.imread(img_path)
                if image is None: continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # ОПТИМИЗАЦИЯ: используем 0 вместо 1, чтобы ускорить поиск в 4 раза
                rects = self.detector(gray, 0)
                
                selected_rect = None
                if len(rects) > 0:
                    gt_center = landmarks.mean(axis=0)
                    best_dist = float('inf')
                    for r in rects:
                        r_center = np.array([(r.left() + r.right())/2, (r.top() + r.bottom())/2])
                        dist = np.linalg.norm(gt_center - r_center)
                        if dist < best_dist:
                            best_dist = dist
                            selected_rect = (r.left(), r.top(), r.right(), r.bottom())
                
                if selected_rect is None:
                    # Если dlib не справился, берем квадрат по точкам разметки
                    x1, y1 = landmarks.min(axis=0).astype(int)
                    x2, y2 = landmarks.max(axis=0).astype(int)
                    selected_rect = (int(x1), int(y1), int(x2), int(y2))
                
                self.preprocessed_data.append({
                    'img_path': img_path,
                    'bbox': selected_rect,
                    'landmarks': landmarks
                })

        # 3. Сохранение результата в кэш
        print(f"--- Сохранение кэша в {cache_path} ---")
        cache_to_save = []
        for item in self.preprocessed_data:
            cache_to_save.append({
                'img_path': item['img_path'],
                'bbox': item['bbox'],
                'landmarks': item['landmarks'].tolist() # numpy -> list для JSON
            })
        
        with open(cache_path, 'w') as f:
            json.dump(cache_to_save, f)
        
        # После сохранения возвращаем landmarks в numpy (для текущей сессии)
        for item in self.preprocessed_data:
            if not isinstance(item['landmarks'], np.ndarray):
                item['landmarks'] = np.array(item['landmarks'], dtype=np.float32)

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        item = self.preprocessed_data[idx]
        image = cv2.imread(item['img_path'])
        if image is None: # Защита от битых файлов
             return torch.zeros((3, self.img_size, self.img_size)), torch.zeros(136)
             
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]
        
        x1, y1, x2, y2 = item['bbox']
        landmarks = item['landmarks'].copy()

        # Применяем Padding
        w_box, h_box = x2 - x1, y2 - y1
        x1_pad = int(max(0, x1 - w_box * self.padding))
        y1_pad = int(max(0, y1 - h_box * self.padding))
        x2_pad = int(min(w_orig, x2 + w_box * self.padding))
        y2_pad = int(min(h_orig, y2 + h_box * self.padding))

        face_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        if face_crop.size == 0: # Защита от пустых кропов
            face_crop = cv2.resize(image, (self.img_size, self.img_size))
            crop_h, crop_w = self.img_size, self.img_size
        else:
            crop_h, crop_w = face_crop.shape[:2]
        
        landmarks[:, 0] = (landmarks[:, 0] - x1_pad) / crop_w
        landmarks[:, 1] = (landmarks[:, 1] - y1_pad) / crop_h

        if self.is_train and np.random.random() > 0.5:
            face_crop = cv2.flip(face_crop, 1)
            landmarks[:, 0] = 1.0 - landmarks[:, 0]
            for left_idx, right_idx in self.FLIP_INDICES:
                landmarks[[left_idx, right_idx]] = landmarks[[right_idx, left_idx]]

        img_tensor = self.transform(face_crop)
        landmarks_tensor = torch.tensor(landmarks.flatten(), dtype=torch.float32)

        return img_tensor, landmarks_tensor

def get_dataloaders(train_path, test_path, batch_size=32):
    train_dataset = FaceLandmarksDataset(train_path, is_train=True)
    test_dataset = FaceLandmarksDataset(test_path, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

from torch.utils.data import DataLoader, ConcatDataset # <-- Добавляем импорт

def get_concat_dataloaders(train_paths, val_path, batch_size=32):
    """
    train_paths: список путей к тренировочным папкам (например, ['300W/train', 'Menpo/train'])
    val_path: путь к папке валидации/теста (например, '300W/test')
    """
    
    # 1. Создаем отдельные датасеты для каждого пути в списке
    train_datasets = []
    for path in train_paths:
        print(f"Подключение тренировочного датасета: {path}")
        dataset = FaceLandmarksDataset(path, is_train=True)
        train_datasets.append(dataset)
        
    # 2. Объединяем их в один БОЛЬШОЙ датасет
    combined_train_dataset = ConcatDataset(train_datasets)
    
    # 3. Создаем датасет для валидации (он обычно один)
    val_dataset = FaceLandmarksDataset(val_path, is_train=False)

    # 4. Создаем DataLoader'ы
    # Для train обязательно shuffle=True, чтобы батчи брали случайные картинки из ОБОИХ датасетов
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Общий размер тренировочной выборки: {len(combined_train_dataset)} картинок.")
    
    return train_loader, val_loader

# # Замени пути на свои
# dataset_300w = FaceLandmarksDataset(root_dir='300W/train', img_size=224)
# dataloader = DataLoader(dataset_300w, batch_size=16, shuffle=True, num_workers=0)

# Проверка одного батча
# for imgs, lbls in dataloader:
#     print(f"Batch images shape: {imgs.shape}") # [16, 3, 224, 224]
#     print(f"Batch labels shape: {lbls.shape}") # [16, 136]
#     break


def show_batch(images, landmarks_batch):
    """
    images: тензор батча [B, 3, 224, 224]
    landmarks_batch: тензор батча [B, 136]
    """
    batch_size = len(images)
    cols = 8
    rows = (batch_size + cols - 1) // cols
    
    plt.figure(figsize=(15, 4 * rows))
    
    for i in range(batch_size):
        img = images[i].numpy().transpose((1, 2, 0))
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        lms = landmarks_batch[i].numpy().reshape(68, 2)
        lms = lms * 224 
        
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.scatter(lms[:, 0], lms[:, 1], s=10, marker='.', c='lime')
        plt.axis('off')
        plt.title(f"Sample {i}")
    
    plt.tight_layout()
    plt.show()


# for imgs, lbls in dataloader:
#     show_batch(imgs, lbls)
#     break