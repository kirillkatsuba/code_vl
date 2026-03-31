import dlib
import cv2
import numpy as np
from data_preprocessing import read_pts_file

detector = dlib.get_frontal_face_detector()
# Загрузка изображения и ориентиров (для одного лица)
img = cv2.imread('300W/test/2353849_1.jpg')
landmarks, n_points = read_pts_file('300W/test/2353849_1.pts')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Убедимся, что landmarks имеет форму (68, 2)
if landmarks.ndim == 3:
    landmarks = landmarks[0]

def draw_landmarks(img, landmarks, offset=(0, 0)):
    out_img = img.copy()
    dx, dy = offset
    for i, (x, y) in enumerate(landmarks):
        cx, cy = int(x - dx), int(y - dy)
        cv2.circle(out_img, (cx, cy), 2, (0, 255, 0), -1)
        cv2.putText(out_img, str(i), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    return out_img

# Детектируем все лица
rects = detector(gray, 1)
print(f"Найдено лиц: {len(rects)}")

# Ищем прямоугольник, который содержит ВСЕ точки landmarks
selected_rect = None
for rect in rects:
    all_inside = True
    for (x, y) in landmarks:
        # Правильная проверка: x – горизонтальная координата, y – вертикальная
        if not (rect.left() <= x <= rect.right() and rect.top() <= y <= rect.bottom()):
            all_inside = False
            break
    if all_inside:
        selected_rect = rect
        break

# Если не нашли подходящий прямоугольник, создаём его по точкам (с отступом)
if selected_rect is None:
    print("Не найден прямоугольник, содержащий все точки. Создаём по точкам с отступом.")
    min_x = int(np.min(landmarks[:, 0])) - 10
    min_y = int(np.min(landmarks[:, 1])) - 10
    max_x = int(np.max(landmarks[:, 0])) + 10
    max_y = int(np.max(landmarks[:, 1])) + 10
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, img.shape[1])
    max_y = min(max_y, img.shape[0])
    selected_rect = dlib.rectangle(left=min_x, top=min_y, right=max_x, bottom=max_y)

# Рисуем точки и рамку на исходном изображении
img_with_landmarks = draw_landmarks(img, landmarks)
x1, y1 = selected_rect.left(), selected_rect.top()
x2, y2 = selected_rect.right(), selected_rect.bottom()
cv2.rectangle(img_with_landmarks, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("Original with landmarks and box", img_with_landmarks)

# Кроп по выбранному прямоугольнику
face_crop = img[y1:y2, x1:x2].copy()
crop_with_landmarks = draw_landmarks(face_crop, landmarks, offset=(x1, y1))
cv2.imshow("Cropped face", crop_with_landmarks)

cv2.waitKey(0)
cv2.destroyAllWindows()