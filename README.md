
# ВАЖНО: в папке app находится прошлая реализация проекта, которая может быть неструктурированна и нефункциональна к моменту сдачи HW1

## Будет изменено в будущем

# Система поддержки принятия решений при управлении скоростным режимом электровоза

Проект разрабатывает интеллектуальную систему, которая с помощью компьютерного зрения определяет текущие координаты поезда, обнаруживает препятствия на пути и предупреждает машиниста о возможных рисках для предотвращения аварийных ситуаций.

---

## 📋 Описание проекта

Система анализирует видеопоток с камеры, установленной на электровозе, с помощью нейронных сетей:

- **Обнаружение километровых и пикетных столбиков** для определения текущей координаты.
- **Распознавание числовых значений** на столбиках через OCR (оптическое распознавание символов).
- **Обнаружение препятствий на пути** (посторонние объекты, животные, автомобили).
- **Сравнение координаты** с данными системы УСАВП.
- **Предупреждение машиниста** о расхождении координат или наличии препятствий.
- **Сбор и мониторинг данных** через PostgreSQL и Grafana.

---

## 🛠️ Стек технологий

- **Computer Vision:** YOLOv8 (обнаружение объектов) (**будет изменено**)
- **OCR:** EasyOCR (распознавание текста на столбиках) (**будет изменено**)
- **Backend:** FastAPI (асинхронное API)
- **Database:** PostgreSQL (хранение координат и аномалий)
- **Monitoring:** Grafana (отображение аномалий в реальном времени)
- **DevOps:** Docker + docker-compose (разворачивание сервисов) (**будет изменено**)
- **Hardware:** ***будет указано позже***

---

## 🧩 Архитектура проекта

```mermaid
---
title: ML architecture flowchart - Система поддержки управления электровозом
---

flowchart TD
    camera[fa:fa-video Camera stream]
    detection_model((YOLO - Object Detection))
    ocr_model((EasyOCR - Text Recognition))
    usavp_data[(USAVP Data)]
    cv_features[[Detected objects: kilometer posts, picket posts, obstacles]]
    ocr_text[Recognized kilometer and picket numbers]
    usavp_coord[USAVP coordinates (km/picket)]
    
    subgraph Analysis
        compare_coord{{Coordinate comparison}}
        detect_anomaly{{Obstacle detection}}
    end

    alert_system{{"Alert System (Warning to Driver)"}}

    monitoring[(PostgreSQL + Grafana - Monitoring and Storage)]

    camera --> detection_model
    detection_model --> cv_features
    detection_model --> ocr_model
    ocr_model --> ocr_text
    usavp_data --> usavp_coord

    cv_features --> compare_coord
    ocr_text --> compare_coord
    usavp_coord --> compare_coord

    cv_features --> detect_anomaly

    compare_coord --> alert_system
    detect_anomaly --> alert_system

    alert_system --> monitoring
```

---

## 🚀 Быстрый старт

1. Клонируйте репозиторий
2. Разверните приложение
**(будет изменено в будущем)**

---

## ⚙️ Основные функции API

| Метод | Эндпоинт | Описание |
|:-----:|:--------:|:-------- |
| `POST` | `/api/data/put_train` | Создать поезд |
| `POST` | `/api/data/start_trip` | Начать поездку |
| `POST` | `/api/data/put_usavp` | Загрузить данные УСАВП |
| `POST` | `/api/data/put_ml` | Загрузить координаты с камеры |
| `GET` | `/api/data/get_statistics` | Получить статистику поездки |

---

## 📈 Метрики качества

- **Точность обнаружения столбиков:** > 98%
- **Точность распознавания текста:** > 90%
- **Обнаружение препятствий:** 100% на тестовых данных
- **Среднее время обработки кадра:** ≤ 100 мс

---

## 🛡️ Безопасность

- Обработка данных локально, без передачи в облако.

---

## 💬 Контакты

**Автор проекта:** erofdmit

**По всем вопросам:** [erofeevdma@gmail.com]  
