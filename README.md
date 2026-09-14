# Automated Sunspot Detection During the Peak of Solar Cycle 25

**An Active Learning Pipeline and Open-Source Benchmark Dataset**

[![Paper](https://img.shields.io/badge/IAC%202026-Paper-blue)](https://iafastro.directory/iac/paper/id/105439/summary/)
[![Dataset](https://img.shields.io/badge/Roboflow-Dataset-purple)](https://universe.roboflow.com/research-8avrk/sunspots-n3q0k)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the code, dataset construction pipeline, and trained model configurations for a human-in-the-loop active learning system that detects sunspots in SDO/HMI continuum imagery during the peak of Solar Cycle 25. The work was presented at the **77th International Astronautical Congress (IAC 2026)**, Antalya, Türkiye.

📄 **Paper:** [IAC-26-A7,IP,16,x105439](https://iafastro.directory/iac/paper/id/105439/summary/)
🗂️ **Dataset (Roboflow Universe):** [sunspots-n3q0k](https://universe.roboflow.com/research-8avrk/sunspots-n3q0k)

---

## Abstract

As Solar Cycle 25 approaches its maximum intensity, accurate real-time monitoring of sunspots is critical for forecasting space weather events, such as geomagnetic storms, that disrupt Earth's power grids and satellite communications. However, the vast data volume produced by the Solar Dynamics Observatory (SDO) and the absence of a labeled dataset for the current solar maximum present a significant bottleneck for traditional manual annotation.

This work introduces a human-in-the-loop active learning pipeline that constructs an open sunspot detection dataset from SDO/HMI Ic_flat_4k continuum imagery, expanding from a manually labeled seed of 1,015 images to **4,552 annotated images** across **six active learning generations**. Using this dataset, we benchmark **YOLOv26** and **RF-DETR** (Nano, Small, and Medium variants of each) across four standardized preprocessing configurations combining resolution and augmentation, training **144 model configurations** in total.

Increasing input resolution from 640×640 to 1024×1024 consistently improves detection performance by 10–15 mAP@50 points across every model and dataset generation tested — a substantially larger effect than model architecture, size, or augmentation. The best-performing configuration, **YOLOv26 Medium at 1024×1024 without augmentation**, achieves **76.9% mAP@50, 80.1% precision, and 74.7% recall**.

To recover small sunspots (pores) lost during standard inference, we additionally implement two Slicing Aided Hyper Inference (SAHI) workflows: a single-model workflow and a two-model consensus ensemble. The ensemble workflow improves mAP@50 by **8.7 points (+19% relative)** and precision by **16.9 points (+40% relative)** over the single-model workflow.

---

## Key Findings

- **Resolution dominates architecture.** Moving from 640×640 to 1024×1024 raises mAP@50 by ~10–15 points across every model and generation tested — far more than switching architecture, scaling model size, or adding augmentation.
- **YOLOv26 outperforms RF-DETR** at every matched configuration and scales more consistently with model size, particularly at 1024×1024.
- **Active learning works, but isn't monotonic.** The dataset grew from a 1,015-image manual seed to 4,552 verified images across six generations with substantially reduced annotation effort. Performance improved from Generation 1 → 2, declined through Generations 3–5, and partially recovered at Generation 6 — a pattern discussed in depth in the paper (Sections 7–8), with candidate explanations including annotation-style drift and cumulative test-set growth.
- **SAHI ensembling recovers precision on tiled inference.** A two-model (YOLOv26 + RF-DETR) consensus workflow over 1024×1024 tiles substantially reduces false positives compared to single-model tiled inference, at the cost of running two models per tile.
- **Recall is the persistent bottleneck.** No configuration in this study exceeds ~75% recall, consistent with the "vanishing pore" problem the SAHI workflows were designed to address.

---

## Repository Structure

```text
.
├── Computer Vision Code/
│   ├── RF-DETR/                  # RF-DETR training / inference scripts
│   └── YOLO/                     # YOLOv26 training / inference scripts
├── Datasets/
│   ├── Version 1/ ... Version 6/ # Dataset generations (see Table 2 in the paper)
│   │   └── Preprocessing V*/     # Resolution × augmentation configs (V1–V4 codes)
├── Research Paper/
│   ├── figures/                  # Pipeline, architecture, and result figures
│   └── Assets/                   # Prediction overlays, supporting assets
├── Videos/                       # Time-lapse renders of 2025 solar activity
├── conduct_predictions_image.py  # Overlay model predictions on a single image
├── create_video.py               # Build time-lapse videos from downloaded frames
├── download_imgs.py              # JSOC scraper for SDO/HMI Ic_flat_4k imagery
├── upload_models.py              # Push trained models to Roboflow
├── .gitattributes                # Git LFS tracking rules
├── .gitignore
└── README.md
```

> Large files (`.pt`, `.pth`, `.zip` dataset exports, video files) are tracked with **Git LFS**. Run `git lfs pull` after cloning to fetch them.

---

## Dataset

The dataset was built through six generations of human-in-the-loop active learning, starting from a manually labeled seed and growing through model-assisted annotation with human correction (the "Oracle step").

| Dataset Version | Total Images | Added Images | Sunspot Annotations | Avg. Sunspots / Image |
|---|---|---|---|---|
| Generation 1 | 1,015 | +1,015 | 23,840 | 23.5 |
| Generation 2 | 1,669 | +654 | 41,325 | 24.8 |
| Generation 3 | 2,349 | +680 | 55,575 | 23.7 |
| Generation 4 | 2,971 | +622 | 82,411 | 27.7 |
| Generation 5 | 3,703 | +732 | 103,657 | 28.0 |
| Generation 6 | 4,552 | +849 | 118,832 | 26.1 |

Images span **1 January – 31 December 2025**, sampled from SDO/HMI Ic_flat_4k continuum imagery at 2-hour intervals to reduce temporal redundancy while capturing solar activity across the full year.

Each dataset generation was processed under four standardized preprocessing configurations:

| Code | Resolution | Augmentation |
|---|---|---|
| V1 | 640×640 | None |
| V2 | 1024×1024 | None |
| V3 | 640×640 | 3× (flip, rotation, crop, blur, noise) |
| V4 | 1024×1024 | 3× (flip, rotation, crop, blur, noise) |

The full dataset, including all generations and preprocessing exports in YOLO and COCO formats, is hosted on Roboflow:

**➡️ [https://universe.roboflow.com/research-8avrk/sunspots-n3q0k](https://universe.roboflow.com/research-8avrk/sunspots-n3q0k)**

Raw source imagery is publicly available from the Joint Science Operations Center (JSOC):
[https://jsoc1.stanford.edu/data/hmi/images/](https://jsoc1.stanford.edu/data/hmi/images/)

---

## Models

Two architectures were benchmarked, chosen to span different levels of technical accessibility for solar-physics researchers:

| Model | Variants | Framework | Training Platform |
|---|---|---|---|
| **YOLOv26** | Nano, Small, Medium | Ultralytics | Kaggle (dual T4 / P100 GPUs) |
| **RF-DETR** | Nano, Small, Medium | Roboflow (no-code) | Roboflow managed platform (A100-SXM4-40GB) |

All models were trained for up to 300 epochs with checkpointing on validation mAP@50 and early stopping after 100 epochs without improvement. Checkpoints were carried forward between active learning generations (best checkpoint from Generation *N* seeds training for Generation *N+1*), with Objects365-pretrained weights used only to initialize Generation 1.

**Best result:** YOLOv26 Medium, 1024×1024, no augmentation, Generation 2 — **76.9% mAP@50 / 80.1% precision / 74.7% recall / 77.3% F1**.

Full per-generation, per-configuration results (144 model runs) are reported in Tables 8–13 of the paper.

---

## High-Resolution Inference (SAHI)

To recover small sunspots (pores) that are lost during standard downscaling, two Slicing Aided Hyper Inference (SAHI) workflows slice full-resolution 4096×4096 images into overlapping 1024×1024 tiles rather than resizing:

1. **Single-model workflow** — YOLOv26 Medium (V2 checkpoint) runs on each tile; detections are stitched with a 0.3 cross-tile IoU threshold.
2. **Ensemble workflow** — YOLOv26 Medium and RF-DETR Medium run independently on each tile; a detection is retained only when both models agree (IoU ≥ 0.5, 2-vote consensus), with confidence averaged across models.

| Metric | Single-model (YOLO alone) | Ensemble (2-vote consensus) | Change |
|---|---|---|---|
| mAP@50 | 46.8% | 55.5% | +8.7 pts (+19%) |
| Precision@50 | 42.0% | 58.9% | +16.9 pts (+40%) |
| Recall@50 | 62.1% | 64.6% | +2.5 pts (+4%) |
| F1@50 | 50.1% | 61.6% | +11.5 pts (+23%) |

Evaluated on the 455-image Generation 6 test split. See Section 4.5 and 6.2 of the paper for full workflow diagrams and discussion.

---

## Reproducing This Work

```bash
# Clone with LFS support
git clone https://github.com/yuseiff/Automated-Sunspot-Detection-Pipeline-using-Active-Learning-for-Solar-Cycle-25.git
cd Automated-Sunspot-Detection-Pipeline-using-Active-Learning-for-Solar-Cycle-25
git lfs pull

```

RF-DETR models were trained via Roboflow's managed platform (see `upload_models.py` for the deployment script); YOLOv26 models were trained locally/on Kaggle via the Ultralytics framework. See `Computer Vision Code/` for the training scripts corresponding to each architecture.

---

## Citation

If you use this dataset, code, or findings in your work, please cite:

```bibtex
@inproceedings{maaod2026sunspot,
  title     = {Automated Sunspot Detection During the Peak of Solar Cycle 25:
               An Active Learning Pipeline and Open-Source Benchmark Dataset},
  author    = {Maaod, Youssef and Eslam, Maryam and Mohamed, Farah and
               El-Shazly, Manar and Abdelkader, Tamer},
  booktitle = {Proceedings of the 77th International Astronautical Congress (IAC 2026)},
  address   = {Antalya, T{\"u}rkiye},
  year      = {2026},
  note      = {Paper IAC-26-A7,IP,16,x105439},
  url       = {https://iafastro.directory/iac/paper/id/105439/summary/}
}
```

Dataset citation:

```bibtex
@misc{sunspot_dataset_2026,
  title  = {Sunspot Detection -- Solar Cycle 25 Dataset},
  author = {{Galala Space Intelligence Group}},
  year   = {2026},
  url    = {https://universe.roboflow.com/research-8avrk/sunspots-n3q0k},
  note   = {Roboflow Universe}
}
```

---

## Limitations

Briefly (see Section 8 of the paper for full discussion):

- Trained on continuum intensity imagery only — no magnetograms, Doppler maps, or spectropolarimetric data.
- Single instrument (HMI), single data product (Ic_flat_4k), single calendar year (2025); generalization to other instruments, cycle phases, or years is untested.
- Later active-learning generations were annotated with progressively heavier model assistance, which may introduce subtle labeling-style drift; the non-monotonic Generation 1→6 performance trend cannot be fully disentangled from cumulative test-set growth using the metrics reported.
- Recall remains the weakest metric throughout (never exceeding ~75%), indicating a persistent share of small pores go undetected regardless of architecture or inference strategy.

## Future Work

- Extend annotation to ~9,000 unlabeled 2021–2024 images to test generalization outside the 2025 solar maximum window.
- Cross-validate against other instruments/observatories (e.g., SOHO/MDI, ground-based archives).
- Incorporate magnetograms and Dopplergrams for physics-informed detection.
- Re-implement RF-DETR training outside Roboflow's managed pipeline for full local reproducibility.
- Evaluate all generations on a common, independently verified test subset with a dedicated inter-annotator agreement study.

---

## Team

| Name | Role |
|---|---|
| Youssef Maaod | Corresponding author |
| Maryam Eslam El-Hossary | Co-author |
| Farah Mohamed | Co-author |
| Manar El-Shazly | Supervisor |
| Tamer Abdelkader | Supervisor |

Faculty of Computer Science and Engineering, Galala University, Suez, Egypt.

---

## License

RF-DETR models: Apache License 2.0. See individual model/checkpoint directories for applicable licenses.

## Acknowledgments

Data provided by NASA's Solar Dynamics Observatory (SDO) / Helioseismic and Magnetic Imager (HMI) team, accessed via the Joint Science Operations Center (JSOC), Stanford University. Dataset annotation and model training infrastructure provided by Roboflow.
