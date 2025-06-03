# файл: custom_inference/yolo_inference.py

import cv2
import numpy as np
from ultralytics import YOLO

def load_model(weights_path: str, device: str = 'cpu'):
    """
    Загружает локальный .pt-файл (например, yolov12m.pt или best.pt) через класс YOLO.
    Возвращает объект модели, готовый к инференсу.

    Args:
        weights_path (str): путь до .pt-файла с весами.
        device (str): 'cpu' или 'cuda' (если хотите на GPU). По умолчанию 'cpu'.
    """
    # Создаем экземпляр модели, передавая путь к вашему .pt
    model = YOLO(weights_path)
    # Переводим на нужное устройство
    model.model.to(device)
    return model


def inference(model, image: np.ndarray, 
              conf_threshold: float = 0.25, 
              iou_threshold: float = 0.45,
              max_det: int = 1000):
    """
    Запускает инференс YOLO-модели (Ultralytics) на одном кадре.
    Возвращает список детекций в формате:
        [
            {'class_id': int, 'confidence': float, 'box': [x_min, y_min, width, height]},
            ...
        ]

    Args:
        model: объект, возвращённый load_model().
        image (np.ndarray): BGR-кадр (OpenCV).
        conf_threshold (float): минимальный порог confidence (0.0–1.0).
        iou_threshold (float): порог NMS (0.0–1.0).
        max_det (int): максимальное число детекций, которые модель отдаст.
    """
    # Ultralytics-секвенс (Onnx/PyTorch) сам ресайзит и нормализует изображение, 
    # если передать ему numpy-массив BGR или RGB. Но лучше сразу в RGB.
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Запускаем инференс. Возвращается список Results (здесь нас интересует первый элемент).
    results = model.predict(
        source=img_rgb,          # можно передать numpy BGR/RGB
        conf=conf_threshold, 
        iou=iou_threshold, 
        max_det=max_det
    )

    # results – это список (len==1), берем первый (для одной картинки)
    res = results[0]

    # res.boxes.xyxy  – shape=(N, 4) в формате [x1, y1, x2, y2]
    # res.boxes.conf  – shape=(N,)  (tensor с уверенность)
    # res.boxes.cls   – shape=(N,)  (tensor с id класса)

    boxes_xyxy = res.boxes.xyxy.cpu().numpy()         # numpy array (N, 4)
    confidences = res.boxes.conf.cpu().numpy()        # numpy array (N,)
    class_ids = res.boxes.cls.cpu().numpy().astype(int)  # numpy array (N,)

    detections = []
    for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confidences, class_ids):
        w = float(x2 - x1)
        h = float(y2 - y1)
        detections.append({
            'class_id': int(cls),
            'confidence': float(conf),
            'box': [float(x1), float(y1), w, h]
        })

    return detections
