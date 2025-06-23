import argparse
import cv2

from custom_inference.yolo_inference import load_model, inference as yolo_inference
from custom_inference.ocr_inference import load_model_and_converter, predict_text, get_device

YOLO_WEIGHTS = "custom_inference/models/yolo/best.pt"
OCR_WEIGHTS = "custom_inference/models/ocr/best.pth"


class Opt:
    imgH = 32
    imgW = 100
    input_channel = 3
    Prediction = "CTC"
    character = "0123456789"


def annotate(frame, detections, texts):
    """Draw bounding boxes and recognized text on the frame."""
    for det, text in zip(detections, texts):
        x1, y1, w, h = det["box"]
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame


def run(video_source: str, fps: int = 1) -> None:
    """Run inference and display results frame by frame."""
    if video_source.isdigit():
        video_source = int(video_source)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {video_source}")

    yolo_model = load_model(YOLO_WEIGHTS)
    device = get_device(use_cuda=False)
    ocr_model, converter = load_model_and_converter(Opt(), OCR_WEIGHTS, device)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(frame_rate // fps), 1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            detections = yolo_inference(yolo_model, frame)
            texts = []
            for det in detections:
                x1, y1, w, h = det["box"]
                crop = frame[int(y1) : int(y1 + h), int(x1) : int(x1 + w)]  # noqa: E203
                text = predict_text(ocr_model, converter, crop, Opt())
                texts.append(text)
            frame = annotate(frame, detections, texts)
        cv2.imshow("Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display inference results frame by frame")
    parser.add_argument("--video", required=True, help="Path to video file or camera index")
    parser.add_argument("--fps", type=int, default=1, help="Inference frames per second")
    args = parser.parse_args()

    run(args.video, args.fps)
