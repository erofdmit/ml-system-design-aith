# streamlit_ocr.py
import os
import cv2
import streamlit as st

CROPS_DIR            = "./crops"
NEW_CROPS_DIR        = "./new_crops"
SUPERRES_OUTPUT_DIR  = os.path.join(NEW_CROPS_DIR, "superres")
THRESH_OUTPUT_DIR    = os.path.join(NEW_CROPS_DIR, "threshold")
OCR_CSV_PATH         = "./ocr_data.csv"
SUPERRES_MODEL_DIR   = "../superres_models"
SUPERRES_MODEL_FILE  = os.path.join(SUPERRES_MODEL_DIR, "EDSR_x4.pb")

def load_superres_model(model_path: str):
    """
    Load OpenCV DNN Super-Resolution model (EDSR ×2).
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    # Even though the file is named “EDSR_x4.pb”, we choose ×2 here
    sr.setModel("edsr", 2)
    return sr

def append_ocr_csv(crop_filename: str, ocr_text: str):
    """
    Append one row to ocr_data.csv: filename,words
    """
    header = "filename,words\n"
    if not os.path.exists(OCR_CSV_PATH):
        with open(OCR_CSV_PATH, "w") as f:
            f.write(header)
    with open(OCR_CSV_PATH, "a") as f:
        # Remove newlines/commas so CSV stays well‐formed
        safe_text = ocr_text.replace("\n", " ").replace(",", " ")
        f.write(f"{crop_filename},{safe_text}\n")

def main():
    st.set_page_config(page_title="OCR Annotation", layout="wide")
    st.title("✍️ OCR Annotation for Crops")

    # ─── ENSURE OUTPUT DIRECTORIES EXIST ─────────────────────────────────────────
    os.makedirs(SUPERRES_OUTPUT_DIR, exist_ok=True)
    os.makedirs(THRESH_OUTPUT_DIR, exist_ok=True)

    # ─── SESSION STATE KEYS ─────────────────────────────────────────────────────
    if "ocr_index" not in st.session_state:
        st.session_state.ocr_index = 0

    if "just_saved_ocr" not in st.session_state:
        st.session_state.just_saved_ocr = False

    # ─── VERIFY CROPS + MODEL EXIST ─────────────────────────────────────────────
    if not os.path.exists(CROPS_DIR):
        st.error(f"No crops directory found at '{CROPS_DIR}'. Run the bbox app first.")
        st.stop()

    if not os.path.exists(SUPERRES_MODEL_FILE):
        st.error(f"Super-Resolution model not found at '{SUPERRES_MODEL_FILE}'.")
        st.stop()

    sr = load_superres_model(SUPERRES_MODEL_FILE)

    all_crops = sorted([
        f for f in os.listdir(CROPS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    total_crops = len(all_crops)

    if total_crops == 0:
        st.info("No crop images found for OCR annotation.")
        st.stop()

    idx = st.session_state.ocr_index

    # ─── IF WE’RE DONE ────────────────────────────────────────────────────────────
    if idx >= total_crops:
        st.success("🎉 All OCR annotations completed!")
        st.stop()

    crop_filename = all_crops[idx]
    crop_path = os.path.join(CROPS_DIR, crop_filename)

    # ─── SHOW “JUST SAVED” BANNER ONCE ────────────────────────────────────────────
    st.markdown(f"**Crop {idx + 1} / {total_crops}**  •  `{crop_filename}`")
    if st.session_state.just_saved_ocr:
        st.success("✅ OCR saved. Moving on…")
        st.session_state.just_saved_ocr = False

    # ─── LOAD RAW IMAGE, SKIP IF UNREADABLE ──────────────────────────────────────
    raw_crop = cv2.imread(crop_path)
    if raw_crop is None:
        st.warning(f"⚠️ Could not read image {crop_filename}. Skipping automatically.")
        st.session_state.ocr_index += 1
        st.stop()  # Next run shows the next crop

    # ─── SUPER-RES UP-SAMPLING & THRESHOLD ────────────────────────────────────────
    upscaled_crop = sr.upsample(raw_crop)
    gray = cv2.cvtColor(upscaled_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ─── SAVE PROCESSED IMAGES TO new_crops ───────────────────────────────────────
    superres_out_path = os.path.join(SUPERRES_OUTPUT_DIR, crop_filename)
    thresh_out_path   = os.path.join(THRESH_OUTPUT_DIR, crop_filename)
    cv2.imwrite(superres_out_path, upscaled_crop)
    cv2.imwrite(thresh_out_path, thresh)

    h_raw, w_raw = raw_crop.shape[:2]
    h_up,  w_up  = upscaled_crop.shape[:2]

    st.write(
        f"• Raw crop size: {w_raw}×{h_raw} px  |  "
        f"Upscaled (×2) size: {w_up}×{h_up} px"
    )
    st.write("The images below use fixed widths so they’re easier to inspect in a narrow sidebar.")

    # ─── DISPLAY: RAW | UPSCALED | THRESHOLDED (SMALLER) ─────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    with col1:
        st.subheader("Raw Crop")
        st.image(
            cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB),
            width=300,
            caption=f"Raw: {w_raw}×{h_raw} px"
        )
    with col2:
        st.subheader("Upscaled Crop (SR ×2)")
        st.image(
            cv2.cvtColor(upscaled_crop, cv2.COLOR_BGR2RGB),
            width=300,
            caption=f"Upscaled: {w_up}×{h_up} px"
        )
    with col3:
        st.subheader("Thresholded + Inverted")
        st.image(
            thresh,
            width=300,
            caption="Binary threshold (inverted)"
        )

    st.markdown("---")
    st.markdown("### Enter OCR Text or Skip This Crop")

    # ─── USE A FORM TO BATCH TEXT + BUTTON EVENTS ─────────────────────────────────
    with st.form(key=f"ocr_form_{idx}"):
        ocr_in = st.text_input(
            "Number / Text for OCR:",
            key=f"ocr_text_{idx}"
        )

        save_col, skip_col = st.columns(2, gap="small")
        with save_col:
            submitted_save = st.form_submit_button("✅ Save & Next")
        with skip_col:
            submitted_skip = st.form_submit_button("→ Skip")

    # ─── HANDLE FORM SUBMISSION ───────────────────────────────────────────────────
    if 'submitted_save' in locals() and submitted_save:
        text = ocr_in.strip()
        if text:
            # 1) Write non‐empty text to CSV
            append_ocr_csv(crop_filename, text)
            st.session_state.just_saved_ocr = True
        else:
            # Warn and stay on this crop if they try to save blank
            st.warning("No text entered. Please type something or click Skip.")
            st.stop()

        # 2) Clear out the old text‐input state so it can't reappear later
        if f"ocr_text_{idx}" in st.session_state:
            st.session_state.pop(f"ocr_text_{idx}")

        # 3) Increment index so next rerun shows the next crop
        st.session_state.ocr_index += 1
        st.stop()

    if 'submitted_skip' in locals() and submitted_skip:
        # 1) Clear any typed text so it doesn’t persist if they revisit
        if f"ocr_text_{idx}" in st.session_state:
            st.session_state.pop(f"ocr_text_{idx}")

        # 2) Move on without writing to CSV
        st.session_state.just_saved_ocr = False
        st.session_state.ocr_index += 1
        st.stop()

if __name__ == "__main__":
    main()
