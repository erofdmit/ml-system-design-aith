import os
import sys
import importlib
from collections import OrderedDict
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .deep_text_recognition.model import Model
from .deep_text_recognition.utils import CTCLabelConverter, AttnLabelConverter

def get_device(use_cuda: bool = True) -> torch.device:
    """Return GPU if available (and requested), else CPU."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_converter(
    opt,
    weights_path: str,
    device: torch.device = None
) -> tuple[nn.Module, object]:
    """
    Load Model(opt) with weights and return (model.eval(), converter).
    Strips 'module.' prefixes. Attaches metadata for debugging.
    """
    # 1) Device
    device = device or get_device()
    # 2) Instantiate and load weights
    model = Model(opt).to(device)
    base_dir = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(models_dir, weights_path)
    state = torch.load(weights_path, map_location=device)
    # 3) Strip 'module.' from keys
    new_state = OrderedDict()
    for k, v in state.items():
        nk = k.removeprefix("module.") if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state)
    model.eval()
    # 4) Converter
    if opt.Prediction == "CTC":
        converter = CTCLabelConverter(opt.character)
    elif opt.Prediction == "Attn":
        converter = AttnLabelConverter(opt.character)
    else:
        raise ValueError(f"Unknown Prediction: {opt.Prediction}")
    # 5) Attach metadata
    model._weights_path = weights_path
    model._opt = opt
    return model, converter


def predict_text(
    model: nn.Module,
    converter,
    frame: np.ndarray,
    opt
) -> str:
    """
    OCR inference on a BGR frame. Supports CTC and Attn,
    prints debug indices and manual mapping.
    """
    device = next(model.parameters()).device
    print(f"[OCR DEBUG] opt.Prediction={opt.Prediction}, imgH={opt.imgH}, imgW={opt.imgW}, chars={len(opt.character)}")
    # BGR->RGB->PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert('L' if opt.input_channel==1 else 'RGB')
    # preprocess
    prep = transforms.Compose([
        transforms.Resize((opt.imgH, opt.imgW)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] if opt.input_channel==1 else [0.5,0.5,0.5],
            std=[0.5] if opt.input_channel==1 else [0.5,0.5,0.5]
        )
    ])
    inp = prep(pil).unsqueeze(0).to(device)
    # forward
    with torch.no_grad():
        if opt.Prediction == 'Attn':
            text_input = torch.zeros((1, opt.batch_max_length), dtype=torch.long, device=device)
            sos = getattr(converter, 'sos_token', 0)
            text_input[:,0] = sos
            out = model(inp, text_input, False)
        else:
            out = model(inp)
    # decode
    if opt.Prediction == 'CTC':
        logits = out.log_softmax(2)
        _, preds = logits.max(2)
        preds = preds.view(1, -1)
        lengths = torch.IntTensor([preds.size(1)]).to(device)
    else:
        _, preds = out.max(2)
        preds = preds.view(1, -1)
        lengths = torch.IntTensor([opt.batch_max_length]).to(device)
    # debug indices
    idxs = preds.squeeze(0).cpu().tolist()
    print(f"[OCR DEBUG] preds_index: {idxs}")
    manual = ''.join(opt.character[i] if i < len(opt.character) else '?' for i in idxs)
    print(f"[OCR DEBUG] manual_mapping: {manual}")
    # converter.decode
    results = converter.decode(preds, lengths)
    text = results[0]
    if opt.Prediction=='Attn':
        text = text.replace('[s]','')
    return text



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python inference_module.py <config_module> <weights.pth> <image_path>")
        sys.exit(1)
    cfg_mod, wpath, img_path = sys.argv[1:]
    spec = importlib.import_module(cfg_mod)
    opt = getattr(spec, "opt", spec.Opt())
    model, converter = load_model_and_converter(opt, wpath)
    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to load image:", img_path)
        sys.exit(1)
    result = predict_text(model, converter, frame, opt)
    print("Recognized text:", result)
