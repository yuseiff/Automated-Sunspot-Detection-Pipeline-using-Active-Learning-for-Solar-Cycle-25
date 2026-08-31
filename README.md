# Automated Sunspot Detection Pipeline using Active Learning for Solar Cycle 25

---

## Abstract

This project presents a **research-grade, end-to-end system for automated sunspot detection** based on **Active Learning**, **high-resolution inference**, and **SAHI-based slicing workflows**, applied to **SDO/HMI Ic_flat_4k continuum data** during **Solar Cycle 25**.

The primary challenge addressed in this work is the **loss of small sunspots (pores)** when high-resolution solar images are processed using conventional downscaling-based object detection pipelines. To overcome this limitation, the proposed system operates directly on **full-disk solar images (~4096×4096 pixels)** using adaptive image slicing rather than resizing. A **Human-in-the-Loop Active Learning framework** is employed to iteratively construct a large, high-quality, cycle-specific dataset while minimizing manual annotation effort.

The pipeline benchmarks **YOLOv12** and **RF-DETR** models across multiple dataset and preprocessing versions, demonstrating that **input resolution has a stronger impact on detection performance than model architecture**, particularly for small-scale solar features. The final system produces reliable, up-to-date sunspot detections and continuously improves as new Solar Cycle 25 data becomes available.

---

## Demo Videos

**Workflow 1 – Single-Model SAHI**
  
![Workflow 1 Video](Videos/workflow1_demo.gif)

**Workflow 2 – Ensemble SAHI**
  
![Workflow 2 Video](Videos/workflow2_demo.gif)

---

## 1. Introduction

Sunspots are regions of intense magnetic activity on the solar photosphere and represent the primary precursors to major solar eruptive events such as solar flares and coronal mass ejections (CMEs). Their number, size, spatial distribution, and morphological complexity are key indicators of the Sun’s activity level and are fundamental inputs to **space weather forecasting systems**.

During **solar maximum**, sunspot emergence rates increase significantly and sunspot groups exhibit highly complex and rapidly evolving structures. Under these conditions, delayed or incomplete detection directly impacts the reliability of space weather predictions, with potential consequences for satellite operations, GNSS systems, aviation, astronaut safety, and terrestrial power grids.

NASA’s **Solar Dynamics Observatory (SDO)** provides continuous, high-resolution observations of the Sun. However, the scale of data produced by SDO far exceeds the capacity of manual analysis. While machine learning offers a natural solution, its effectiveness depends critically on the availability of **high-quality, representative labeled datasets**, which are currently lacking for **Solar Cycle 25**, particularly near its peak.

---

## 2. Data Source and Acquisition

- **Mission:** Solar Dynamics Observatory (SDO)  
- **Instrument:** Helioseismic and Magnetic Imager (HMI)  
- **Product:** Ic_flat_4k (Continuum)  
- **Native Resolution:** 4096 × 4096 pixels  

Solar images are retrieved from JSOC servers using a dedicated Python script and sampled at approximately **12 images per day**.

---

## 3. Preprocessing Strategy

Small sunspots (pores) occupy only a few pixels in the original images. Aggressive downscaling causes these features to disappear, resulting in low recall despite high apparent precision. Multiple preprocessing versions (V1–V10) were evaluated, confirming that **high-resolution inputs (1024×1024)** are essential.

---

## 4. Active Learning Pipeline (Human-in-the-Loop)

**Figure 1. Active Learning Pipeline**

![Figure 1](Research Paper/figures/active_learning_pipeline.png)

The dataset is expanded iteratively through manual seeding, model training, auto-labeling, human correction, and retraining.

---

## 5. Iteration History and Dataset Growth

**Figure 2. Iteration History and Progress Cycle**

![Figure 2](Research Paper/figures/iteration_history.png)

| Generation | Dataset Size |
|-----------|-------------|
| Gen 1 | 1,015 |
| Gen 2 | 1,145 |
| Gen 3 | 1,669 |
| Current | 3,965 |

---

## 6. High-Resolution Inference with SAHI

Instead of resizing, images are sliced into overlapping tiles that preserve native resolution. Tile sizes are selected dynamically:
- 640×640 for YOLO
- 1024×1024 for RF-DETR

---

## 7. Inference Workflows

### 7.1 Workflow 1 – Single-Model SAHI

**Figure 3. Workflow 1 Architecture**

![Figure 3](Research Paper/figures/workflow1_sahi.png)

---

### 7.2 Workflow 2 – Ensemble SAHI

**Figure 4. Workflow 2 Architecture**

![Figure 4](Research Paper/figures/workflow2_ensemble.png)

---

## 8. Models and Benchmarking

Best configuration:
- **Model:** RF-DETR Medium  
- **Resolution:** 1024×1024  

| Metric | Value |
|------|------|
| mAP@50 | 74.65% |
| Precision | 80.6% |
| Recall | 72.0% |

---

## 9. Qualitative Results

**Figure 5. Best RF-DETR Result**

![Figure 5](Research Paper/figures/rfdetr_best.png)

**Figure 6. Workflow Comparison**

![Figure 6](Research Paper/figures/workflow_comparison.png)

---

## 10. Project Structure

```text
.
├── Computer Vision Code/
│   ├── RF-DETR/
│   └── YOLO/
├── Datasets/
│   ├── Version 1/
│   ├── Version 2/
│   └── Version 3/
├── Research Paper/
├── Videos/
├── conduct_predictions_image.py
├── create_video.py
├── download_imgs.py
├── .gitattributes
├── .gitignore
└── README.md
```

---

## 11. Future Work

- Complete annotation of remaining Solar Cycle 25 data
- Extend dataset to historical cycles
- Improve ensemble strategies
- Add temporal sunspot tracking

---

## Team

| Name |
|------|
| Youssef Husseiny Fathy Maaod |
| Maryam Eslam Elhossary |
| Farah Mohamed |

---

## Licences
RF-DETR Models: apache-2.0 license