"""
Sliced object detection workflow using a pretrained detection model.

Mirrors the workflow graph:

    Inputs (image)
        -> Image Slicer
        -> Object Detection Model (pretrained, YOLO26 or RF-DETR)
        -> Detections Stitch
        -> Bounding Box Visualization
        -> Outputs (detections_stitch, model_predictions, image)

Slicing an image into tiles before running detection (and stitching the
per-tile boxes back into full-image coordinates) is the standard trick for
finding small objects -- like sunspots -- that would otherwise be missed or
blurred together when the full image is downscaled to the model's input size.

Install:
    pip install ultralytics supervision opencv-python tqdm
"""

import os

import cv2
import numpy as np
import supervision as sv
from supervision.metrics import F1Score, MeanAveragePrecision, Precision, Recall
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Fixed paths / config -- edit these directly instead of passing CLI args.
# ---------------------------------------------------------------------------
IMAGE_PATH = r"Computer Vision Code\Workflow\prediction\image.jpg"
OUTPUT_DIR = r"Computer Vision Code\Workflow\prediction"
_image_stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{_image_stem}.jpg")

YOLO_MODEL_PATH = r"Computer Vision Code\YOLOv26\YOLO26_Dataset Version2\Preprocessing2\YOLO26_M_V2_V2\weights\best.pt"          # pretrained YOLO26 checkpoint
RFDETR_MODEL_PATH = "rfdetr_weights.pth"  # pretrained RF-DETR checkpoint (placeholder)

# Which model to run: "yolo" or "rfdetr"
MODEL_TYPE = "yolo"

SLICE_WH = (1024, 1024)
OVERLAP_RATIO_WH = (0.2, 0.2)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# Which pipeline to run when the script is executed directly:
# "single"   -> run_workflow() on IMAGE_PATH / OUTPUT_PATH (as before)
# "evaluate" -> evaluate_workflow() against the labeled test set below
RUN_MODE = "evaluate"

# ---------------------------------------------------------------------------
# Test dataset (YOLO format export, e.g. from Roboflow) used for evaluation.
# Expected layout:
#   TEST_IMAGES_DIR/*.jpg
#   TEST_LABELS_DIR/*.txt
#   TEST_DATA_YAML        (data.yaml with the class names)
# ---------------------------------------------------------------------------
TEST_IMAGES_DIR = r"Datasets\Final Dataset 4k\Sunspots.yolo26\test\images"
TEST_LABELS_DIR = r"Datasets\Final Dataset 4k\Sunspots.yolo26\test\labels"
TEST_DATA_YAML = r"Datasets\Final Dataset 4k\Sunspots.yolo26\data.yaml"


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_yolo26(model_path: str = YOLO_MODEL_PATH):
    """
    Loads a pretrained YOLO26 model for inference (no training).

    Returns:
        model: the raw ultralytics YOLO model
        predict_fn: callable(image_slice) -> sv.Detections, for use as the
                    InferenceSlicer callback
        class_names: dict mapping class_id -> class name
    """
    model = YOLO(model_path)

    def predict_fn(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice, conf=CONF_THRESHOLD, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    class_names = model.model.names
    return model, predict_fn, class_names


def load_RF_DETR(model_path: str = RFDETR_MODEL_PATH):
    """
    Placeholder for loading a pretrained RF-DETR model for inference.
    To be implemented -- should return (model, predict_fn, class_names)
    with the same signature as load_yolo26.
    """
    raise NotImplementedError("load_RF_DETR is not implemented yet.")


def load_model(model: str = "yolo"):
    """Dispatches to the right loader based on `model`: 'yolo' or 'rfdetr'."""
    if model == "yolo":
        return load_yolo26()
    elif model == "rfdetr":
        return load_RF_DETR()
    else:
        raise ValueError(f"Unknown model type: {model!r}. Use 'yolo' or 'rfdetr'.")


# ---------------------------------------------------------------------------
# Shared slicer construction (Image Slicer + Detections Stitch node)
# ---------------------------------------------------------------------------
def build_slicer(predict_fn) -> sv.InferenceSlicer:
    """
    sv.InferenceSlicer cuts the image into overlapping tiles, runs
    `predict_fn` on each one, and stitches all the per-tile detections
    back into full-image coordinates (de-duplicating overlaps via NMS).
    Newer supervision versions (>=0.27) take a pixel-based overlap_wh
    instead of a ratio, so we convert OVERLAP_RATIO_WH to pixels here.
    """
    overlap_wh = (
        int(SLICE_WH[0] * OVERLAP_RATIO_WH[0]),
        int(SLICE_WH[1] * OVERLAP_RATIO_WH[1]),
    )
    return sv.InferenceSlicer(
        callback=predict_fn,
        slice_wh=SLICE_WH,
        overlap_wh=overlap_wh,
        iou_threshold=IOU_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Workflow: single image
# ---------------------------------------------------------------------------
def run_workflow(model: str = MODEL_TYPE) -> dict:
    # --- Inputs node ---------------------------------------------------
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    # --- Object Detection Model node -----------------------------------
    _, predict_fn, class_names = load_model(model)

    # --- Image Slicer + Detections Stitch node --------------------------
    slicer = build_slicer(predict_fn)
    detections: sv.Detections = slicer(image)

    # --- Bounding Box Visualization node -----------------------------
    labels = [
        f"{class_names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    cv2.imwrite(OUTPUT_PATH, annotated)

    # --- Outputs node -------------------------------------------------
    return {
        "detections_stitch": detections,
        "model_predictions": detections,
        "image": annotated,
    }


# ---------------------------------------------------------------------------
# Evaluation: run the full workflow over a labeled YOLO-format test set and
# compute mAP@50, precision, recall, and F1 (all at IoU 0.50) by comparing
# stitched predictions against ground-truth annotations.
# ---------------------------------------------------------------------------
def evaluate_workflow(model: str = MODEL_TYPE) -> dict:
    # Loads images + YOLO .txt annotations + class names from data.yaml
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=TEST_IMAGES_DIR,
        annotations_directory_path=TEST_LABELS_DIR,
        data_yaml_path=TEST_DATA_YAML,
    )

    _, predict_fn, _ = load_model(model)
    slicer = build_slicer(predict_fn)

    map_metric = MeanAveragePrecision()
    precision_metric = Precision()
    recall_metric = Recall()
    f1_metric = F1Score()

    progress_bar = tqdm(dataset, total=len(dataset), desc="Evaluating", unit="img")
    for image_path, image, ground_truth in progress_bar:
        predictions: sv.Detections = slicer(image)

        map_metric.update(predictions, ground_truth)
        precision_metric.update(predictions, ground_truth)
        recall_metric.update(predictions, ground_truth)
        f1_metric.update(predictions, ground_truth)

        progress_bar.set_postfix(
            pred=len(predictions),
            gt=len(ground_truth),
        )

    map_result = map_metric.compute()
    precision_result = precision_metric.compute()
    recall_result = recall_metric.compute()
    f1_result = f1_metric.compute()

    metrics = {
        "mAP@50": map_result.map50,
        "mAP@50-95": map_result.map50_95,
        "precision@50": precision_result.precision_at_50,
        "recall@50": recall_result.recall_at_50,
        "f1@50": f1_result.f1_50,
    }

    print("\n=== Evaluation Results (IoU 0.50) ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    if RUN_MODE == "evaluate":
        evaluate_workflow(model=MODEL_TYPE)
    else:
        outputs = run_workflow(model=MODEL_TYPE)
        print(f"Detections found: {len(outputs['detections_stitch'])}")
        print(f"Annotated image saved to: {OUTPUT_PATH}")