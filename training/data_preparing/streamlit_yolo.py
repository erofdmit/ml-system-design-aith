
import os
import cv2
import pandas as pd
import numpy as np
import streamlit as st

MODEL_PATH         = "./" 
CSV_FILENAME       = "./bboxes/video_4_bboxes.csv"
VIDEO_FILENAME     = "./videos/test_video_4.mp4"

IMAGES_DIR         = "images"
LABELS_DIR         = "labels"
CROPS_DIR          = "crops"
ANNOTATIONS_CSV    = "annotations.csv"

CONFIDENCE_FILTER  = 0.75

@st.cache_data
def load_detection_csv(path_to_csv: str) -> pd.DataFrame:
    """
    Load the detection CSV into a DataFrame, validate required columns,
    filter by confidence > CONFIDENCE_FILTER, and sort by frame → bbox_id.
    """
    df = pd.read_csv(path_to_csv)
    required = {"video", "frame", "bbox_id", "x1", "y1", "x2", "y2", "confidence", "class_id"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Detection CSV is missing columns: {missing}")

    df["frame"]      = df["frame"].astype(int)
    df["bbox_id"]    = df["bbox_id"].astype(int)
    df["x1"]         = df["x1"].astype(int)
    df["y1"]         = df["y1"].astype(int)
    df["x2"]         = df["x2"].astype(int)
    df["y2"]         = df["y2"].astype(int)
    df["confidence"] = df["confidence"].astype(float)

    df = df[df["confidence"] > CONFIDENCE_FILTER].reset_index(drop=True)

    df = df.sort_values(["frame", "bbox_id"]).reset_index(drop=True)
    return df

def ensure_folders_exist():
    """Ensure that images/, labels/, and crops/ exist."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)

def save_frame_image_once(frame_img: np.ndarray, frame_num: int):
    """
    Save full frame to images/frame_<frame_num>.jpg once per frame.
    """
    if frame_num not in st.session_state.processed_frames:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        path = os.path.join(IMAGES_DIR, f"frame_{frame_num}.jpg")
        cv2.imwrite(path, frame_img)
        st.session_state.processed_frames.add(frame_num)

def append_yolo_label(frame_num: int,
                      class_id: int,
                      x1: int, y1: int, x2: int, y2: int,
                      frame_width: int, frame_height: int):
    """
    Append one line to labels/frame_<frame_num>.txt in YOLO format.
    Return normalized coords for logging.
    """
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    bb_w     = (x2 - x1)
    bb_h     = (y2 - y1)

    x_center_norm = x_center / frame_width
    y_center_norm = y_center / frame_height
    width_norm    = bb_w / frame_width
    height_norm   = bb_h / frame_height

    os.makedirs(LABELS_DIR, exist_ok=True)
    label_path = os.path.join(LABELS_DIR, f"frame_{frame_num}.txt")
    with open(label_path, "a") as f:
        f.write(
            f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} "
            f"{width_norm:.6f} {height_norm:.6f}\n"
        )
    return x_center_norm, y_center_norm, width_norm, height_norm

def append_annotations_csv(frame_num: int,
                           bbox_id: int,
                           class_label: str,
                           class_id: int,
                           x1: int, y1: int, x2: int, y2: int,
                           x_center_norm: float,
                           y_center_norm: float,
                           width_norm: float,
                           height_norm: float):
    """
    Append one row to annotations.csv with all details (excluding OCR).
    """
    header = ("frame,bbox_id,class_label,class_id,"
              "x1,y1,x2,y2,"
              "x_center_norm,y_center_norm,width_norm,height_norm\n")
    write_header = not os.path.exists(ANNOTATIONS_CSV)
    with open(ANNOTATIONS_CSV, "a") as f:
        if write_header:
            f.write(header)
        f.write(
            f"{frame_num},{bbox_id},{class_label},{class_id},"
            f"{x1},{y1},{x2},{y2},"
            f"{x_center_norm:.6f},{y_center_norm:.6f},"
            f"{width_norm:.6f},{height_norm:.6f}\n"
        )

def main():
    st.set_page_config(page_title="BBox Annotation (No OCR)", layout="wide")
    st.title("📌 Bounding‐Box Verification / Annotation")
    st.markdown(f"**Only showing detections with confidence > {CONFIDENCE_FILTER}**")
    
    csv_path   = os.path.join(MODEL_PATH, CSV_FILENAME)
    video_path = os.path.join(MODEL_PATH, VIDEO_FILENAME)
    try:
        df = load_detection_csv(csv_path)
    except Exception as e:
        st.error(f"⚠️ Could not load CSV: {e}")
        st.stop()

    total_detections = len(df)
    st.write(f"Total detections after filtering: {total_detections}")


    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "processed_frames" not in st.session_state:
        st.session_state.processed_frames = set()
    if "just_saved" not in st.session_state:
        st.session_state.just_saved = False

    if st.session_state.current_index >= total_detections:
        st.success("🎉 All detections processed!")
        st.write("• Frames → images/")
        st.write("• YOLO labels → labels/")
        st.write("• Crops (raw) → crops/")
        st.write("• Master CSV → annotations.csv")
        return


    idx = st.session_state.current_index
    row = df.iloc[idx]
    frame_num  = int(row["frame"])
    bbox_id    = int(row["bbox_id"])
    x1, y1     = int(row["x1"]), int(row["y1"])
    x2, y2     = int(row["x2"]), int(row["y2"])
    confidence = float(row["confidence"])

    st.markdown(
        f"**Detection {idx + 1} / {total_detections}**  •  "
        f"Frame #{frame_num}, BBox #{bbox_id}, Confidence {confidence:.2f}"
    )
    if st.session_state.just_saved:
        st.success("✅ Annotation saved. Moving on…")

    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        st.error(f"❌ Could not open video at {video_path}")
        st.stop()

    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num < 0 or frame_num >= total_frames:
        st.warning(f"⚠️ Frame {frame_num} out of range (video has {total_frames}). Skipping.")
        st.session_state.current_index += 1
        return

    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = vc.read()
    vc.release()
    if not ret:
        st.warning(f"⚠️ Could not read frame {frame_num}. Skipping.")
        st.session_state.current_index += 1
        return

    frame_with_box = frame.copy()
    cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (255, 0, 0), 2)

    raw_crop = frame[y1:y2, x1:x2]
    if raw_crop.size == 0:
        st.warning("⚠️ Crop is empty (zero area). Skipping.")
        st.session_state.current_index += 1
        return

    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        st.image(
            cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB),
            caption=f"Full frame #{frame_num} (BBox #{bbox_id})",
            use_column_width=True
        )

    with col2:
        st.markdown("#### Jump to detection")
        jump_to = st.number_input(
            label="Row (1-based)",
            min_value=1,
            max_value=total_detections,
            value=st.session_state.current_index + 1,
            step=1,
            key="jump_input"
        )
        if st.button("Go", key="jump_btn"):
            st.session_state.current_index = int(jump_to) - 1
            st.session_state.just_saved = False

        st.markdown("---")
        st.markdown("### Annotation")
        with st.form(key="annotation_form"):
            choice = st.radio(
                label="Class or skip:",
                options=["skip", "kilometer_post", "picket_post"],
                index=0,
                horizontal=True
            )

            save_next = st.form_submit_button("✅ Save & Next")

            if save_next:
                if choice == "skip":
                    st.session_state.just_saved = True
                    st.session_state.current_index += 1
                else:
                    ensure_folders_exist()

                    save_frame_image_once(frame, frame_num)

                    H, W = frame.shape[:2]
                    class_map = {"picket_post": 0, "kilometer_post": 1}
                    class_id  = class_map[choice]
                    x_cn, y_cn, w_n, h_n = append_yolo_label(
                        frame_num=frame_num,
                        class_id=class_id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        frame_width=W, frame_height=H
                    )

                    os.makedirs(CROPS_DIR, exist_ok=True)
                    crop_filename = f"frame_{frame_num}_bbox_{bbox_id}.png"
                    crop_path = os.path.join(CROPS_DIR, crop_filename)
                    cv2.imwrite(crop_path, raw_crop)

                    append_annotations_csv(
                        frame_num=frame_num,
                        bbox_id=bbox_id,
                        class_label=choice,
                        class_id=class_id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        x_center_norm=x_cn,
                        y_center_norm=y_cn,
                        width_norm=w_n,
                        height_norm=h_n
                    )

                    st.session_state.just_saved = True
                    st.session_state.current_index += 1

                return 
if __name__ == "__main__":
    main()
