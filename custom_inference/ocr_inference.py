import sys
import importlib
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from deep_text_recognition.model import Model
from deep_text_recognition.utils import CTCLabelConverter, AttnLabelConverter  # <-- ваши декодеры


def get_device(use_cuda=True):
    """Возвращает устройство (GPU, если доступно, иначе CPU)."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_converter(opt, weights_path: str, device=None):
    """
    Загружает модель Model с весами и создаёт соответствующий LabelConverter.

    Args:
        opt: объект с атрибутами конфигурации модели (imgH, imgW, input_channel, Prediction, character и др.).
        weights_path (str): путь к .pth файлу с state_dict.
        device (torch.device или str, optional): устройство для модели. Если None, выбирается автоматически.

    Returns:
        model (torch.nn.Module)  – модель в режиме eval() на указанном устройстве.
        converter           – экземпляр CTCLabelConverter или AttnLabelConverter в зависимости от opt.Prediction.
    """
    # 1) Определяем устройство
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    # 2) Загружаем модель
    model = Model(opt).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) Создаём конвертер символов ←→ индексы
    if opt.Prediction == 'CTC':
        converter = CTCLabelConverter(opt.character)
    elif opt.Prediction == 'Attn':
        converter = AttnLabelConverter(opt.character)
    else:
        raise ValueError(f"Unknown Prediction mode: {opt.Prediction}")

    return model, converter


def predict_text(model: nn.Module, converter, frame: 'np.ndarray', opt) -> str:
    """
    Выполняет предсказание текста для одного кадра и сразу возвращает строку.

    Args:
        model (torch.nn.Module): загруженная модель в режиме eval().
        converter: экземпляр CTCLabelConverter или AttnLabelConverter.
        frame (np.ndarray): кадр в формате BGR (OpenCV) dtype=uint8.
        opt: объект конфигурации с теми же параметрами, что использованы для обучения.

    Returns:
        str: распознанная строка.
    """
    device = next(model.parameters()).device

    # 1) Конвертируем BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # 2) Если модель ожидала одноканальное изображение, конвертим в 'L'
    if opt.input_channel == 1:
        pil_img = pil_img.convert('L')
    else:
        pil_img = pil_img.convert('RGB')

    # 3) Препроцессинг: resize до (imgH, imgW), ToTensor, Normalize
    transform_list = [
        transforms.Resize((opt.imgH, opt.imgW)),  # Resize ожидает (H, W)
        transforms.ToTensor()
    ]
    if opt.input_channel == 1:
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    preprocess = transforms.Compose(transform_list)

    input_tensor = preprocess(pil_img)              # shape: [C, H, W]
    input_batch  = input_tensor.unsqueeze(0).to(device)  # shape: [1, C, H, W]

    # 4) Прямой проход через модель
    with torch.no_grad():
        output = model(input_batch)
        # output.shape:
        #   - для CTC: [1, seq_len, num_classes]
        #   - для Attn: [1, max_length, num_classes]

    # 5) Декодинг
    if opt.Prediction == 'CTC':
        # 5.1) Для CTC: сначала берем argmax по последней размерности (num_classes)
        #      в результате получим тензор size=[1, seq_len] с индексами наиболее вероятных символов.
        logits = output.log_softmax(2)          # [1, seq_len, num_classes]
        _, preds_index = logits.max(2)          # preds_index: [1, seq_len]
        preds_index = preds_index.view(1, -1)   # всё ещё shape=[1, seq_len]

        # 5.2) Длины предсказания: seq_len именно такое, какое модель выдала.
        seq_length = preds_index.size(1)
        length_for_pred = torch.IntTensor([seq_length]).to(device)  # [1]

        # 5.3) converter.decode ждёт:
        #      - text_index: LongTensor shape [batch_size, seq_len]
        #      - length:    IntTensor shape [batch_size], каждый элемент – “сколько символов учитывать”
        preds_str_list = converter.decode(preds_index, length_for_pred)
        # converter.decode возвращает список строк длины batch_size (здесь batch_size=1)
        preds_str = preds_str_list[0]

    elif opt.Prediction == 'Attn':
        # 5.1) Для Attn: output уже logits размера [1, max_length, num_classes]
        _, preds_index = output.max(2)         # preds_index: [1, max_length]
        preds_index = preds_index.view(1, -1)  # всё ещё [1, max_length]

        # 5.2) Длина предсказания: обычно это opt.batch_max_length (максимальная длина + 1 для [s])
        length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)

        preds_str_list = converter.decode(preds_index, length_for_pred)
        preds_str = preds_str_list[0]

    else:
        raise ValueError(f"Unsupported Prediction mode: {opt.Prediction}")

    return preds_str


if __name__ == "__main__":
    # Пример использования из командной строки:
    # python inference_module.py <config_module> <weights.pth> <image_path>
    if len(sys.argv) < 4:
        print("Использование: python inference_module.py <config_module> <путь_к_весам> <путь_к_изображению>")
        sys.exit(1)

    config_module = sys.argv[1]
    weights_path  = sys.argv[2]
    image_path    = sys.argv[3]

    # Динамически импортируем модуль с opt
    spec = importlib.import_module(config_module)
    opt = spec.opt if hasattr(spec, 'opt') else spec.Opt()

    # 1) Загружаем модель + конвертер
    model, converter = load_model_and_converter(opt, weights_path)

    # 2) Читаем изображение (BGR)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Не удалось загрузить изображение {image_path}")
        sys.exit(1)

    # 3) Получаем распознанную строку
    recognized_text = predict_text(model, converter, frame, opt)
    print("Recognized text:", recognized_text)
