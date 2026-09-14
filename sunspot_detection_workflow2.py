"""
Two-model ensemble object detection workflow (YOLO26 + RF-DETR via Roboflow API).

Mirrors the workflow graph:

    Inputs (image)
        -> Image Slicer
            -> Object Detection Model (YOLO26, local .pt checkpoint)   -> Detections Stitch (model)
            -> Object Detection Model (RF-DETR, Roboflow hosted API)   -> Detections Stitch (model_1)
        -> Detections Consensus   (merges the two stitched detection sets)
        -> Bounding Box Visualization
        -> Outputs (predictions, all_properties, image)

Both models see the same tiling scheme (same SLICE_WH / OVERLAP_RATIO_WH),
each producing its own full-image stitched detections independently. The
consensus step then decides which detections to keep based on whether both
models agree (see CONSENSUS_REQUIRED_VOTES below).

Install:
    pip install ultralytics supervision opencv-python tqdm inference-sdk

Set your Roboflow API key as an environment variable before running:
    setx ROBOFLOW_API_KEY "your_key_here"      (Windows, new terminal after)
    export ROBOFLOW_API_KEY="your_key_here"    (macOS/Linux)
"""

import csv
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from supervision.metrics import F1Score, MeanAveragePrecision, Precision, Recall
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Fixed paths / config -- edit these directly instead of passing CLI args.
# ---------------------------------------------------------------------------
IMAGE_PATH = r"Raw Data\2025\07\31\20250731_060000_Ic_flat_4k.jpg"
OUTPUT_DIR = r"Computer Vision Code\Workflow\prediction"
_image_stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{_image_stem}.jpg")

# --- Model 1: local YOLO26 checkpoint --------------------------------------
YOLO_MODEL_PATH = r"Computer Vision Code\YOLOv26\YOLO26_Dataset Version2\Preprocessing2\YOLO26_M_V2_V2\weights\best.pt"

# --- Model 2: RF-DETR, hosted on Roboflow (called over the network) --------
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")  # set this env var, don't hardcode the key
ROBOFLOW_MODEL_ID = "sunspots-n3q0k/70"

# Best config found for YOLO26 via the earlier evaluation runs -- see
# sunspot_detection_workflow.py for the experiment history.
SLICE_WH = (1024, 1024)
OVERLAP_RATIO_WH = (0.2, 0.2)
OP_CONF_THRESHOLD = 0.5   # per-model confidence filter, applied before consensus
IOU_THRESHOLD = 0.3       # cross-tile stitching threshold (per model)
THREAD_WORKERS = 1        # keep at 1 for the YOLO slicer -- see note in build_slicer()

# --- Detections Consensus settings -----------------------------------------
# How much two detections (one per model) must overlap to be treated as the
# same real object.
CONSENSUS_IOU_THRESHOLD = 0.5
# 1 -> keep a detection if EITHER model found it (higher recall, more false positives)
# 2 -> keep a detection only if BOTH models found it (higher precision, may miss real spots)
CONSENSUS_REQUIRED_VOTES = 2
# How to combine confidence scores for a detection both models agreed on.
CONSENSUS_CONFIDENCE_AGG = "mean"  # "mean" or "max"

CLASS_NAMES = {0: "Sunspot"}  # single-class task; used for both models' visualization labels

# Which pipeline to run when the script is executed directly:
# "single"   -> run_workflow() on IMAGE_PATH / OUTPUT_PATH
# "evaluate" -> evaluate_workflow() against the labeled test set below
RUN_MODE = "evaluate"

# ---------------------------------------------------------------------------
# Test dataset (YOLO format export, e.g. from Roboflow) used for evaluation.
# ---------------------------------------------------------------------------
TEST_IMAGES_DIR = r"Datasets\Final Dataset 4k\Sunspots.yolo26\test\images"
TEST_LABELS_DIR = r"Datasets\Final Dataset 4k\Sunspots.yolo26\test\labels"
TEST_DATA_YAML = r"Datasets\Final Dataset 4k\Sunspots.yolo26\data.yaml"

LOG_DIR = r"Computer Vision Code\Workflow\prediction\evaluation_logs"


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_yolo26(model_path: str = YOLO_MODEL_PATH, conf_threshold: float = OP_CONF_THRESHOLD):
    """
    Loads a pretrained YOLO26 model for inference (no training).

    Returns:
        model: the raw ultralytics YOLO model
        predict_fn: callable(image_slice) -> sv.Detections
        class_names: dict mapping class_id -> class name
    """
    model = YOLO(model_path)

    # Warm up: ultralytics lazily builds/fuses its internal predictor on the
    # first predict() call, which is not thread-safe if called concurrently.
    _dummy = np.zeros((SLICE_WH[1], SLICE_WH[0], 3), dtype=np.uint8)
    model.predict(_dummy, conf=conf_threshold, verbose=False)

    def predict_fn(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice, conf=conf_threshold, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    return model, predict_fn, CLASS_NAMES


def load_RF_DETR(
    api_url: str = ROBOFLOW_API_URL,
    api_key: str = ROBOFLOW_API_KEY,
    model_id: str = ROBOFLOW_MODEL_ID,
    conf_threshold: float = OP_CONF_THRESHOLD,
):
    """
    Loads an RF-DETR model hosted on Roboflow, called over HTTP for each tile.

    Note: this makes one network request per tile, per image. For a 4096x4096
    image sliced at 1024x1024 with 20% overlap that's ~25 API calls per image
    -- expect this to be much slower than the local YOLO26 model, and to
    consume Roboflow API credits/rate limits accordingly.

    Returns:
        client: the raw InferenceHTTPClient
        predict_fn: callable(image_slice) -> sv.Detections
        class_names: dict mapping class_id -> class name
    """
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY is not set. Set it as an environment variable "
            "before running (see the module docstring) -- never hardcode it "
            "in the script."
        )

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    try:
        # Newer inference_sdk versions (>=1.5.0) support explicit header-based
        # auth via api_key_transport. Older installed versions don't have
        # this field on InferenceConfiguration -- fall back gracefully since
        # api_key is already passed to the client above either way.
        client = client.configure(InferenceConfiguration(api_key_transport="header"))
    except TypeError:
        print(
            "Note: installed inference_sdk version doesn't support "
            "api_key_transport (likely older than 1.5.0) -- continuing "
            "with default auth transport instead."
        )

    def predict_fn(image_slice: np.ndarray) -> sv.Detections:
        result = client.infer(image_slice, model_id=model_id)
        detections = sv.Detections.from_inference(result)
        # Roboflow API doesn't take a per-call confidence filter here, so
        # apply it client-side to match the local model's behavior.
        return detections[detections.confidence >= conf_threshold]

    return client, predict_fn, CLASS_NAMES


def load_model(model: str = "yolo", conf_threshold: float = OP_CONF_THRESHOLD):
    """Dispatches to the right loader based on `model`: 'yolo' or 'rfdetr'."""
    if model == "yolo":
        return load_yolo26(conf_threshold=conf_threshold)
    elif model == "rfdetr":
        return load_RF_DETR(conf_threshold=conf_threshold)
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
        thread_workers=THREAD_WORKERS,
    )


# ---------------------------------------------------------------------------
# Detections Consensus node
# ---------------------------------------------------------------------------
def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between every box in A and every box in B. Shape (len_a, len_b)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter_area = inter_w * inter_h

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return np.where(union_area > 0, inter_area / union_area, 0.0)


def detections_consensus(
    detections_a: sv.Detections,
    detections_b: sv.Detections,
    iou_threshold: float = CONSENSUS_IOU_THRESHOLD,
    required_votes: int = CONSENSUS_REQUIRED_VOTES,
    confidence_agg: str = CONSENSUS_CONFIDENCE_AGG,
):
    """
    Merges two stitched detection sets from independent models into one
    consensus set.

    Matching: greedy, highest-IoU-first pairing between A and B boxes above
    `iou_threshold`. Matched pairs become one consensus detection (2 votes);
    unmatched detections get 1 vote each and are kept only if
    `required_votes` <= 1.

    Returns:
        consensus_detections: sv.Detections
        all_properties: list of dicts, one per kept detection, with vote
                         count and per-model confidences (for the "output" /
                         all_properties node in the diagram)
    """
    boxes_a, boxes_b = detections_a.xyxy, detections_b.xyxy
    conf_a = detections_a.confidence if detections_a.confidence is not None else np.ones(len(boxes_a))
    conf_b = detections_b.confidence if detections_b.confidence is not None else np.ones(len(boxes_b))
    class_a = detections_a.class_id if detections_a.class_id is not None else np.zeros(len(boxes_a), dtype=int)
    class_b = detections_b.class_id if detections_b.class_id is not None else np.zeros(len(boxes_b), dtype=int)

    iou_matrix = _box_iou_matrix(boxes_a, boxes_b)

    matched_a, matched_b = set(), set()
    pairs = []  # list of (a_idx, b_idx)

    # Greedy matching, strongest overlap first.
    flat_order = np.dstack(np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape))[0]
    for a_idx, b_idx in flat_order:
        a_idx, b_idx = int(a_idx), int(b_idx)
        if a_idx in matched_a or b_idx in matched_b:
            continue
        if iou_matrix[a_idx, b_idx] < iou_threshold:
            break  # sorted descending -- everything after this is also below threshold
        pairs.append((a_idx, b_idx))
        matched_a.add(a_idx)
        matched_b.add(b_idx)

    out_boxes, out_conf, out_class, all_properties = [], [], [], []

    for a_idx, b_idx in pairs:
        box = (boxes_a[a_idx] + boxes_b[b_idx]) / 2.0
        conf = (
            float(np.mean([conf_a[a_idx], conf_b[b_idx]]))
            if confidence_agg == "mean"
            else float(max(conf_a[a_idx], conf_b[b_idx]))
        )
        out_boxes.append(box)
        out_conf.append(conf)
        out_class.append(int(class_a[a_idx]))  # single-class task -- classes assumed to agree
        all_properties.append({
            "vote_count": 2,
            "source_models": ["yolo", "rfdetr"],
            "individual_confidences": [float(conf_a[a_idx]), float(conf_b[b_idx])],
        })

    if required_votes <= 1:
        for a_idx in range(len(boxes_a)):
            if a_idx in matched_a:
                continue
            out_boxes.append(boxes_a[a_idx])
            out_conf.append(float(conf_a[a_idx]))
            out_class.append(int(class_a[a_idx]))
            all_properties.append({
                "vote_count": 1,
                "source_models": ["yolo"],
                "individual_confidences": [float(conf_a[a_idx])],
            })
        for b_idx in range(len(boxes_b)):
            if b_idx in matched_b:
                continue
            out_boxes.append(boxes_b[b_idx])
            out_conf.append(float(conf_b[b_idx]))
            out_class.append(int(class_b[b_idx]))
            all_properties.append({
                "vote_count": 1,
                "source_models": ["rfdetr"],
                "individual_confidences": [float(conf_b[b_idx])],
            })

    if out_boxes:
        consensus_detections = sv.Detections(
            xyxy=np.array(out_boxes, dtype=np.float32),
            confidence=np.array(out_conf, dtype=np.float32),
            class_id=np.array(out_class, dtype=int),
        )
    else:
        consensus_detections = sv.Detections.empty()

    return consensus_detections, all_properties


# ---------------------------------------------------------------------------
# Workflow: single image
# ---------------------------------------------------------------------------
def run_workflow2() -> dict:
    # --- Inputs node ---------------------------------------------------
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    # --- Object Detection Model (x2) + Detections Stitch (x2) -----------
    _, predict_fn_yolo, _ = load_model("yolo", conf_threshold=OP_CONF_THRESHOLD)
    _, predict_fn_rfdetr, _ = load_model("rfdetr", conf_threshold=OP_CONF_THRESHOLD)

    slicer_yolo = build_slicer(predict_fn_yolo)
    slicer_rfdetr = build_slicer(predict_fn_rfdetr)

    print("Running YOLO26 (local)...")
    detections_yolo: sv.Detections = slicer_yolo(image)
    print(f"  -> {len(detections_yolo)} detections")

    print("Running RF-DETR (Roboflow API)...")
    detections_rfdetr: sv.Detections = slicer_rfdetr(image)
    print(f"  -> {len(detections_rfdetr)} detections")

    # --- Detections Consensus node --------------------------------------
    consensus_detections, all_properties = detections_consensus(detections_yolo, detections_rfdetr)
    print(f"Consensus: {len(consensus_detections)} detections "
          f"(required_votes={CONSENSUS_REQUIRED_VOTES})")

    # --- Bounding Box Visualization node -----------------------------
    labels = [
        f"{CLASS_NAMES.get(class_id, 'obj')} {confidence:.2f}"
        for class_id, confidence in zip(consensus_detections.class_id, consensus_detections.confidence)
    ]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = box_annotator.annotate(scene=image.copy(), detections=consensus_detections)
    annotated = label_annotator.annotate(scene=annotated, detections=consensus_detections, labels=labels)

    cv2.imwrite(OUTPUT_PATH, annotated)

    # --- Outputs node -------------------------------------------------
    return {
        "output2": consensus_detections,     # detections_consensus.predictions
        "output": all_properties,            # detections_consensus.all_properties
        "consensus": annotated,              # bounding_box_visualization.image
    }


# ---------------------------------------------------------------------------
# Evaluation: run the ensemble over a labeled YOLO-format test set and
# compute mAP@50, precision, recall, and F1 for the consensus output.
# ---------------------------------------------------------------------------
def evaluate_workflow2() -> dict:
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=TEST_IMAGES_DIR,
        annotations_directory_path=TEST_LABELS_DIR,
        data_yaml_path=TEST_DATA_YAML,
    )

    _, predict_fn_yolo, _ = load_model("yolo", conf_threshold=OP_CONF_THRESHOLD)
    _, predict_fn_rfdetr, _ = load_model("rfdetr", conf_threshold=OP_CONF_THRESHOLD)
    slicer_yolo = build_slicer(predict_fn_yolo)
    slicer_rfdetr = build_slicer(predict_fn_rfdetr)

    map_metric = MeanAveragePrecision()
    precision_metric = Precision()
    recall_metric = Recall()
    f1_metric = F1Score()

    os.makedirs(LOG_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(LOG_DIR, f"{run_id}_ensemble_config.json")
    metrics_csv_path = os.path.join(LOG_DIR, f"{run_id}_ensemble_metrics.csv")

    run_config = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "workflow": "ensemble (yolo26 + rfdetr via roboflow api)",
        "yolo_model_path": YOLO_MODEL_PATH,
        "rfdetr_model_id": ROBOFLOW_MODEL_ID,
        "slice_wh": list(SLICE_WH),
        "overlap_ratio_wh": list(OVERLAP_RATIO_WH),
        "op_conf_threshold": OP_CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "consensus_iou_threshold": CONSENSUS_IOU_THRESHOLD,
        "consensus_required_votes": CONSENSUS_REQUIRED_VOTES,
        "consensus_confidence_agg": CONSENSUS_CONFIDENCE_AGG,
        "test_images_dir": TEST_IMAGES_DIR,
        "test_labels_dir": TEST_LABELS_DIR,
        "test_data_yaml": TEST_DATA_YAML,
        "num_images": len(dataset),
        "final_metrics": {},
    }

    def save_config():
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)

    save_config()
    print(f"Run config:          {config_path}")
    print(f"Per-image metrics:   {metrics_csv_path}")

    fieldnames = [
        "image_index", "image_path", "elapsed_sec",
        "num_yolo", "num_rfdetr", "num_consensus", "num_ground_truth",
        "mAP@50", "mAP@50-95", "precision@50", "recall@50", "f1@50",
    ]
    csv_file = open(metrics_csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    metrics: dict = {}
    start_time = time.time()
    progress_bar = tqdm(dataset, total=len(dataset), desc="Evaluating ensemble", unit="img")

    try:
        for idx, (image_path, image, ground_truth) in enumerate(progress_bar, start=1):
            detections_yolo = slicer_yolo(image)
            detections_rfdetr = slicer_rfdetr(image)
            consensus_detections, _ = detections_consensus(detections_yolo, detections_rfdetr)

            map_metric.update(consensus_detections, ground_truth)
            precision_metric.update(consensus_detections, ground_truth)
            recall_metric.update(consensus_detections, ground_truth)
            f1_metric.update(consensus_detections, ground_truth)

            map_result = map_metric.compute()
            precision_result = precision_metric.compute()
            recall_result = recall_metric.compute()
            f1_result = f1_metric.compute()

            metrics = {
                "mAP@50": round(float(map_result.map50), 4),
                "mAP@50-95": round(float(map_result.map50_95), 4),
                "precision@50": round(float(precision_result.precision_at_50), 4),
                "recall@50": round(float(recall_result.recall_at_50), 4),
                "f1@50": round(float(f1_result.f1_50), 4),
            }

            writer.writerow({
                "image_index": idx,
                "image_path": image_path,
                "elapsed_sec": round(time.time() - start_time, 2),
                "num_yolo": len(detections_yolo),
                "num_rfdetr": len(detections_rfdetr),
                "num_consensus": len(consensus_detections),
                "num_ground_truth": len(ground_truth),
                **metrics,
            })
            csv_file.flush()

            progress_bar.set_postfix(
                yolo=len(detections_yolo),
                rfdetr=len(detections_rfdetr),
                consensus=len(consensus_detections),
                gt=len(ground_truth),
                mAP50=metrics["mAP@50"],
            )

        run_config["status"] = "completed"

    except KeyboardInterrupt:
        run_config["status"] = "interrupted"
        print("\nEvaluation interrupted -- partial results have already been saved.")

    finally:
        csv_file.close()
        run_config["finished_at"] = datetime.now().isoformat(timespec="seconds")
        run_config["final_metrics"] = metrics
        save_config()

    print("\n=== Ensemble Evaluation Results ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"\nFull config + final metrics: {config_path}")
    print(f"Per-image progress log:      {metrics_csv_path}")

    return metrics


if __name__ == "__main__":
    if RUN_MODE == "evaluate":
        evaluate_workflow2()
    else:
        outputs = run_workflow2()
        print(f"Consensus detections: {len(outputs['output2'])}")
        print(f"Annotated image saved to: {OUTPUT_PATH}")