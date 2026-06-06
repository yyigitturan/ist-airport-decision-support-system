# Multimodal Flight Post-Terminal Duration Prediction
### via Cross-Modality Adaptation of Large Language Models and Self-Supervised Trajectory Representation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/LLM-Qwen2.5--0.5B-purple" />
  <img src="https://img.shields.io/badge/Airport-LTFM%20Istanbul-teal" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/🤗-ATSCC%20Model-yellow" />
</p>

> **Undergraduate Final Project** · Department of Statistics, Hacettepe University  
> **Author:** Yaşar Yiğit Turan · **Supervisor:** Prof. Dr. Serpil Aktaş Altunay

---

## Table of Contents

- [Overview](#overview)
- [Problem Definition](#problem-definition)
- [Data Pipeline](#1-data-pipeline)
- [Label Engineering](#2-label-engineering)
- [ATSCC — Self-Supervised Trajectory Representation](#3-atscc--self-supervised-trajectory-representation)
- [Multimodal LLM Regressor](#4-multimodal-llm-regressor)
- [Results](#results)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Overview

Flight delays at Istanbul Airport (LTFM) — one of Europe's busiest aviation hubs — impose significant operational costs and cascade effects throughout the air traffic network. Accurate prediction of arrival duration within controlled airspace is a key enabler for proactive Air Traffic Flow Management (ATFM).

This work proposes a **multimodal deep learning framework** that predicts *post-terminal duration*: the time an arriving aircraft spends inside the Terminal Maneuvering Area (TMA), defined as the period from TMA entry (~120 km radius) to touchdown. The framework combines:

1. **ATSCC** — a self-supervised ADS-B trajectory encoder trained with a contrastive objective, producing segment-aware 256-dimensional flight embeddings without any labels.
2. **Multimodal LLM Regressor** — cross-modality adaptation of a frozen Qwen2.5-0.5B language model that fuses trajectory embeddings (focusing flight + active co-flying aircraft + prior landed aircraft) with structured flight plan text and real-time METAR/TAF weather prompts.

This approach is inspired by and extends the cross-modality adaptation paradigm introduced by Phisannupawong et al. (KAIST, 2024) for flight delay prediction.


---

## Problem Definition

<p align="center">
  <img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/8834fd1c-fb77-4bd3-a3bd-53dffc834899" />
</p>


Given a snapshot of the airspace at observation time *t* — including the focusing flight's partial ADS-B trajectory, co-flying active flights, and recently landed prior flights — the model predicts the **remaining time (minutes) until touchdown**.

This framing differs from classical delay prediction in two important ways:
- The label is **continuous airspace duration**, not a binary delay flag or discrete category.
- The input includes **real-time airspace context** (who else is in the TMA right now), not just the flight's own metadata.

---

### 1. Data Pipeline

The system ingests data from three source types through a **medallion lakehouse architecture**:

<p align="center">
  <img width="881" height="494" alt="image" src="https://github.com/user-attachments/assets/60bc5907-971a-41d7-b889-8cb4cb75583a" />
</p>


| Layer | Description |
|---|---|
| **Bronze** | Raw ADS-B trajectory archive + flight information records + aviation weather reports (METAR/TAF). No transformations applied. |
| **Silver** | Cleaned, standardized, and normalized data. Derived columns, data enrichment, anomaly filtering. |
| **Gold** | Feature-ready data with full integrations, aggregations, and business logic. Fed directly into modeling. |

**Source modalities:**
- **ADS-B Trajectory Archive** — position in ENU frame (`e_m`, `n_m`, `u_m`), direction vector (`ux`, `uy`, `uz`), polar coordinates (`r`, `sin_theta`, `cos_theta`), time gap (`delta_t`), data gap flag (`gap_flag`)
- **Flight Information Records** — airline, aircraft type, registration, wake turbulence category, route, origin/destination ICAO codes
- **Aviation Weather Reports** — METAR + TAF strings for LTFM at each observation time

---

### 2. Label Engineering

Ground truth labels (post-terminal duration) are constructed **algorithmically** from raw ADS-B data through a four-step pipeline — no manual annotation required.


<p align="center">
  <img width="524" height="485" alt="image" src="https://github.com/user-attachments/assets/41f8a01e-2f2e-4c88-9ceb-30383c79cb53" width="550"/>
 
</p>

**Step 1 — Find Landing Proxy**  
The radar point closest to the airport near the scheduled arrival time is selected as the proxy landing fix. Search windows expand progressively: ±90 min → ±180 min → ±360 min.

**Step 2 — Verify Approach Trend**  
An approach ratio is computed as the fraction of timesteps showing monotonically decreasing distance to the airport:

```
ratio = (# decreasing distance steps) / (total steps)
```

- 10 min **before** landing: `ratio > 0.60` required
- 10 min **after** landing: `ratio < 0.40` required
- Holding pattern detected: threshold relaxed to `0.45`

**Step 3 — Find TMA Entry Point**  
Searching up to 3 hours prior to the scheduled landing time, the first point crossing the ~120 km TMA boundary is identified. Data gaps exceeding 30 minutes are skipped to ensure trajectory continuity.

**Step 4 — Data Validation**  
All candidate samples must satisfy:
- `25 km ≤ entry_dist ≤ 120 km`
- `landing_dist ≤ 18 km`
- `entry_time ≤ landing_time` (logical sequence enforced)
- Arrival-like descent trend confirmed

Only samples passing all four criteria are forwarded to the modeling stage.

---

### 3. ATSCC — Self-Supervised Trajectory Representation

The **ATC Self-Supervised Contrastive Classifier (ATSCC)** is a causal transformer encoder pre-trained with a Siamese Contrastive Loss objective. It learns segment-aware trajectory representations without any labels by pulling together timesteps belonging to the **same RDP segment** (positive pairs) and pushing apart timesteps from **different segments** (negative pairs).

**Architecture:**

<img width="423" height="345" alt="image" src="https://github.com/user-attachments/assets/6615b65a-cfd8-491b-b9fa-526493510484" />



**Key design decisions:**
- **Causal masking** — each timestep only attends to past context, reflecting the online nature of ATC decision-making
- **RDP segmentation** — Ramer–Douglas–Peucker algorithm identifies geometric turning points, splitting trajectories into semantically meaningful segments used as contrastive labels
- **Random masking** (prob=0.15) and **DropPath** (rate=0.10) during training for regularization
- **L2 normalization** at input and output projections for stable contrastive learning

**Model settings:**

| Parameter | Value |
|---|---|
| Layers | 4 |
| Attention Heads | 8 |
| d_model | 192 |
| d_ff | 768 |
| Embedding Dimension | 256 |
| Max Sequence Length | 256 |
| Temperature | 0.10 |
| Masking Probability | 0.15 |
| Dropout | 0.25 |
| DropPath Rate | 0.10 |
| Effective Batch Size | 64 (32 × 2 grad accum) |
| Learning Rate | 7e-6 |
| Optimizer | AdamW (wd=5e-4) |
| LR Schedule | Warmup (4 ep) + Cosine Annealing |
| Best Epoch | 17 |
| Best Val Loss | 0.6045 |

**Embedding quality** is evaluated by two metrics from representation learning literature:
- **Alignment** (lower is better): mean cosine similarity of same-segment point pairs — measures whether the encoder correctly groups trajectory segments
- **Uniformity** (closer to 0 is better): log-mean of pairwise Gaussian kernel distances — measures whether embeddings are well-distributed across the hypersphere

<p align="center">
  <img width="3153" height="1290" alt="image" src="https://github.com/user-attachments/assets/eb320059-e3dd-4c3a-8627-c7f0ee80da06" width="650"/> 
</p>

The pretrained ATSCC encoder is available on Hugging Face:  
🤗 | **HuggingFace** | [atscc-trajectory-ist-airport-encoder](https://huggingface.co/yyigitturan/atscc-trajectory-ist-airport-encoder) |

---

### 4. Multimodal LLM Regressor

The regression head adapts a **frozen Qwen2.5-0.5B-Instruct** LLM for multimodal duration prediction via a cross-modality bridging mechanism. The LLM backbone is never fine-tuned — only the lightweight adapter layers and regression head are trained.

#### Airspace Scenario Construction

At each observation time *t*, three trajectory embedding streams are constructed from the airspace snapshot using the pretrained ATSCC encoder:


<p align="center">
  <img width="659" height="378" alt="image" src="https://github.com/user-attachments/assets/f96566b8-0608-4ed7-a3a9-15d1c01e948b" />
</p>


#### Full Architecture

<p align="center">
  <img width="725" height="485" alt="image" src="https://github.com/user-attachments/assets/60720296-f81e-4ad1-a56a-b9edbb4a79b1" 
/p>



#### Input Prompts

**Flight Plan Prompt** — structured natural language describing the focusing flight:

```
Current time: <OBS_TIME>.
Actual airspace entry time for flight <CALLSIGN> was <ENTRY_TIME>.
This <HAUL_TYPE> flight operated by <AIRLINE> is scheduled to arrive
at <ARR_TIME> on <DATE>.
It originated from <DEP_AIRPORT> (<DEP_ICAO> / <DEP_IATA>),
dep lat: <DEP_LAT>, lon: <DEP_LON>, alt: <DEP_ALT> ft,
and was headed for Istanbul Airport (<DEST_ICAO> / <DEST_IATA>)
dest lat: <DEST_LAT>, lon: <DEST_LON>, alt: <DEST_ALT> ft.
Aircraft type: <AIRCRAFT_TYPE>. Registration: <REG>.
Wake turbulence category: <WTC>. Total route distance: <DISTANCE> km.
```

**Weather Information Prompt** — raw METAR + TAF strings for LTFM at observation time, providing real-time visibility, wind, cloud ceiling, and precipitation context.

#### Training Configuration

| Setting | Value |
|---|---|
| Base LLM | Qwen2.5-0.5B-Instruct |
| LLM Hidden Dim | 896 |
| Trajectory Dim | 256 |
| Adapter Dropout | 0.15 |
| Head Dropout | 0.15 |
| Loss | MSELoss |
| Learning Rate | 1e-5 |
| Weight Decay | 2e-5 |
| Warmup Ratio | 0.06 |
| Grad Accum Steps | 16 |
| Early Stopping Patience | 3 epochs |
| Target Scaling | StandardScaler (per month) |
| Max Prompt Length | 775 tokens |
| Max Active Flights | 21 |
| Max Prior Flights | 20 |

---

## Results

Monthly out-of-sample prediction performance on the radar-to-landing duration regression task:

| Month | MAE | MSE | RMSE | R² | Adj. R² | SMAPE (%) |
|---|---|---|---|---|---|---|
| 2025-03 | 2.0206 | 7.3706 | 2.7149 | 0.7934 | 0.7933 | 7.31 |
| 2025-06 | **1.1614** | **4.4891** | **2.1187** | **0.9059** | **0.9059** | **5.62** |
| 2025-07 | 1.2102 | 3.5811 | 1.8924 | 0.9140 | 0.9140 | 6.64 |
| 2025-08 | 1.4517 | 5.0792 | 2.2537 | 0.8864 | 0.8864 | 5.50 |
| 2025-09 | 2.3633 | 12.9002 | 3.5917 | 0.5643 | 0.5641 | 7.21 |
| 2025-10 | 1.3894 | 4.1342 | 2.0333 | 0.9012 | 0.9010 | 6.32 |
| 2025-11 | 2.0692 | 8.5511 | 2.9242 | 0.5474 | 0.5467 | 6.61 |
| 2025-12 | 1.6320 | 7.4413 | 2.7279 | 0.7978 | 0.7977 | 5.57 |
| 2026-01 | 2.2300 | 8.1053 | 2.8470 | 0.7071 | 0.7068 | 6.74 |



**Key findings:**
- **Summer months (June–July 2025)** yield the strongest performance (R² > 0.90, MAE < 1.25 min), benefiting from higher and more consistent traffic volume and predictable approach patterns.
- **Autumn/winter months (September, November)** show higher variance, likely due to weather-induced holding events, irregular sequencing, and lower training data volume.
- The model achieves **sub-3-minute RMSE across all evaluated months**, demonstrating practical viability for ATFM decision support.
- July 2025 achieves the best RMSE (1.89 min) while June achieves the best MAE (1.16 min) and R² (0.914).

---


## Dependencies

| Package | Role |
|---|---|
| `torch` | ATSCC training & LLM inference |
| `transformers` + `accelerate` | Qwen2.5-0.5B loading & cross-modality adaptation |
| `pandas` + `pyarrow` | Data lakehouse pipeline & parquet I/O |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing & evaluation metrics |
| `rdp` | Ramer–Douglas–Peucker trajectory segmentation |
| `pymap3d` | ECEF ↔ geodetic coordinate conversion |
| `umap-learn` | Embedding space visualization |
| `sentencepiece` | Tokenizer support for Qwen |
| `tqdm` | Progress bars |

---

## Acknowledgements

This work draws on and extends the cross-modality adaptation framework for flight delay prediction proposed in:

> Phisannupawong, T., Damanik, J. J., & Choi, H.-L. (2024). *Flight Delay Prediction via Cross-Modality Adaptation of Large Language Models and Aircraft Trajectory Representation.* Department of Aerospace Engineering, KAIST, Republic of Korea.

The self-supervised contrastive learning objective for trajectory segmentation is inspired by Siamese network designs in representation learning literature. ADS-B data processing follows [OpenSky Network](https://opensky-network.org/) data format conventions.

---

## Citation

```bibtex
@misc{turan2026atscc,
  title  = {Multimodal Flight Post-Terminal Duration Prediction via
             Cross-Modality Adaptation of Large Language Models and
             Self-Supervised Trajectory Representation},
  author = {Turan, Yaşar Yiğit},
  year   = {2026},
  note   = {Undergraduate Final Project, Department of Statistics,
             Hacettepe University}
}
```

---

## Contact

| | |
|---|---|
| **Email** | yasaryigitturan@gmail.com |
| **LinkedIn** | [yaşar-yiğit-turan](https://www.linkedin.com/in/yaşar-yiğit-turan-/) |
| **HuggingFace** | [atscc-trajectory-ist-airport-encoder](https://huggingface.co/yyigitturan/atscc-trajectory-ist-airport-encoder) |

---

<p align="center">
  <sub>Department of Statistics · Hacettepe University · Ankara, Turkey</sub>
</p>
