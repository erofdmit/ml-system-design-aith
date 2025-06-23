import os
import random
import shutil
from pathlib import Path

path = os.path.abspath('.')
SRC_IMG_DIR   = Path(os.path.join(path, "dataset/yolo/images/train"))
SRC_LABEL_DIR =  Path(os.path.join(path, "dataset/yolo/labels/train"))


DST_IMG_TRAIN =  Path(os.path.join(path, "dataset/yolo/images/train"))
DST_IMG_VAL   =  Path(os.path.join(path, "dataset/yolo/images/val"))
DST_LABEL_TRAIN =  Path(os.path.join(path, "dataset/yolo/labels/train"))
DST_LABEL_VAL   =  Path(os.path.join(path, "dataset/yolo/labels/val"))

split_ratio = 0.8  

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ────────────────────────────────────────────────────────────────────────────

for folder in [DST_IMG_TRAIN, DST_IMG_VAL, DST_LABEL_TRAIN, DST_LABEL_VAL]:
    folder.mkdir(parents=True, exist_ok=True)

all_images = [f for f in SRC_IMG_DIR.iterdir() 
              if f.suffix.lower() in IMG_EXTENSIONS]
random.shuffle(all_images)

n_train = int(len(all_images) * split_ratio)

train_images = all_images[:n_train]
val_images   = all_images[n_train:]

def move_pairs(image_list, img_dest, label_dest):
    for img_path in image_list:
        dst_img_path = img_dest / img_path.name
        shutil.move(str(img_path), str(dst_img_path))

        label_fn = img_path.with_suffix(".txt").name
        src_label = SRC_LABEL_DIR / label_fn
        dst_label = label_dest / label_fn
        if src_label.exists():
            shutil.move(str(src_label), str(dst_label))
        else:
            print(f"Warning: missing label for {img_path.name}")

move_pairs(train_images, DST_IMG_TRAIN, DST_LABEL_TRAIN)
move_pairs(val_images,   DST_IMG_VAL,   DST_LABEL_VAL)

print("Done splitting. Train:", len(train_images), "images. Val:", len(val_images), "images.")
