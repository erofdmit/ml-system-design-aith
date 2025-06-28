import ast
from argparse import Namespace
import os


OCR_OPT_PATH = os.getenv("OCR_OPT_PATH", "custom_inference/models/ocr/opt.txt")

def load_opt_from_txt(path):
    opts = {}
    with open(path, 'r') as f:
        for line in f:
            # skip separators and blank lines
            if line.startswith('---') or not line.strip() or ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            # try to interpret booleans, lists, numbers, etc.
            try:
                val = ast.literal_eval(val)
            except Exception:
                pass
            opts[key] = val
    return Namespace(**opts)

# usage:
opt = load_opt_from_txt(OCR_OPT_PATH)
