# Multimodal Flight Post-Terminal Duration Prediction via Cross-Modality Adaptation of Large Language Models and Self-Supervised Trajectory Representation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/LLM-Qwen2.5--0.5B-purple" />
  <img src="https://img.shields.io/badge/Airport-LTFM%20Istanbul-teal" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

> **Undergraduate Final Project** · Department of Statistics, Hacettepe University  
> **Author:** Yaşar Yiğit Turan · **Supervisor:** Prof. Dr. Serpil Aktaş Altunay

---

## Table of Contents

- [Overview](#overview)
- [Problem Definition](#problem-definition)
- [System Architecture](#system-architecture)
  - [Data Pipeline](#1-data-pipeline)
  - [Label Engineering](#2-label-engineering)
  - [ATSCC — Self-Supervised Trajectory Representation](#3-atscc--self-supervised-trajectory-representation)
  - [Multimodal LLM Regressor](#4-multimodal-llm-regressor)
- [Dataset](#dataset)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Overview

Flight delays at Istanbul Airport (LTFM) — one of Europe's busiest aviation hubs — impose significant operational costs and cascade effects throughout the air traffic network. Accurate prediction of arrival duration within controlled airspace is a key enabler for proactive air traffic flow management (ATFM).

This work proposes a **multimodal deep learning framework** that predicts *post-terminal duration*: the time an arriving aircraft spends inside the Terminal Maneuvering Area (TMA), defined as the period from TMA entry (~120 km radius) to touchdown. The framework combines:

1. **Self-supervised ADS-B trajectory representations** learned via a contrastive objective (ATSCC), and
2. **Cross-modality adaptation of a frozen large language model** (Qwen2.5-0.5B) that fuses trajectory embeddings with structured flight plan and weather (METAR/TAF) prompts.

This approach is inspired by and extends the cross-modality adaptation paradigm introduced by Phisannupawong et al. for flight delay prediction.

---

## Problem Definition

```
TMA Entry (~120 km) ──────────────────────────────► Landing (LTFM)
        │                                                  │
   ENTRY_TIME                                        PROXY_LANDING_TIME
        └──────────── Post-Terminal Duration ─────────────┘
```

Given a snapshot of the airspace at observation time *t* — including the focusing flight's partial trajectory, co-flying active flights, and recently landed prior flights — the model predicts the **remaining time (minutes) until touchdown**.

---

## System Architecture

### 1. Data Pipeline

The system ingests data from three source types through a medallion lakehouse architecture:

| Layer | Description |
|---|---|
| **Bronze** | Raw ADS-B trajectory archive + flight information records + aviation weather reports (METAR/TAF). No transformations applied. |
| **Silver** | Cleaned, standardized, and normalized data. Derived columns, data enrichment, and anomaly filtering. |
| **Gold** | Feature-ready data with full integrations, aggregations, and business logic applied. Fed directly into modeling. |

**Source modalities:**
- ADS-B Trajectory Archive (position: ECEF `e,n,u`; direction: `uₑ,uₙ,uᵤ`; polar coordinates `r,sinθ,cosθ`; time gap `Δt`; gap flag)
- Flight Information Records (airline, aircraft type, registration, wake turbulence category, route, origin/destination ICAO)
- Aviation Weather Reports (METAR + TAF for LTFM)

---

### 2. Label Engineering

Ground truth labels (post-terminal duration) are constructed algorithmically from raw ADS-B data through a four-step pipeline:

**Step 1 — Find Landing Proxy**  
The radar point closest to the airport near the scheduled arrival time is selected as the proxy landing fix (within ±90 min / ±180 min / ±360 min windows).

**Step 2 — Verify Approach Trend**  
An approach ratio is computed as the fraction of steps showing monotonically decreasing distance to the airport:

```
ratio = (# decreasing distance steps) / (total steps)
```

- 10 min before landing: `ratio > 0.60` required
- 10 min after landing: `ratio < 0.40` required
- Holding pattern detected: threshold relaxed to `0.45`

**Step 3 — Find TMA Entry Point**  
Searching up to 3 hours prior to the scheduled landing time, the first point crossing the ~120 km TMA boundary is identified. Data gaps exceeding 30 minutes are skipped.

**Step 4 — Data Validation**  
All candidate samples must satisfy:
- `25 km ≤ entry_dist ≤ 120 km`
- `landing_dist ≤ 18 km`
- `entry_time ≤ landing_time` (logical sequence)
- Arrival-like descent trend confirmed

Only samples passing all four criteria are forwarded to the modeling stage.

---

### 3. ATSCC — Self-Supervised Trajectory Representation

The **ATC Self-Supervised Contrastive Classifier (ATSCC)** is a causal transformer encoder pre-trained with a Siamese Contrastive Loss objective to produce semantically rich, segment-aware trajectory embeddings.

**Architecture:**

```
ADS-B Trajectory Data
        │
  RDP Segmentation ──► assigns Segment ID to each timestep
        │
     Linear + L2 Norm
        │
  Causal Transformer Encoder (4 layers)
  ├── Causal Encoder Layer
  ├── Causal Encoder Layer
  ├── Causal Encoder Layer
  └── Causal Encoder Layer
        │
     Linear + L2 Norm
        │
  Trajectory Embedding  ──────────────────────────────┐
        │                                              │
  Encoder Layer                                        │
  ├── Masked Multi-Head Self-Attention                 │
  │   (DropPath + Layer Norm)                          │
  └── Feed Forward                                     │
      (DropPath + Layer Norm)                          │
        │                                              │
  SNN Contrastive Loss ◄──────── Segment ID ───────────┘
        │
  Backpropagation → Weight Updates
```

**Model Settings:**

| Parameter | Value |
|---|---|
| Layers | 4 |
| Attention Heads | 8 |
| d_model | 192 |
| d_ff | 768 |
| Embedding Dimension | 256 |
| Max Sequence Length | 256 |
| Temperature | 0.10 |
| Masking | 0.15 |
| Dropout | 0.25 |
| DropPath | 0.10 |
| Batch Size | 64 (effective) |
| Learning Rate | 7e-6 |

**Embedding Quality** is evaluated by two metrics:
- **Alignment** (lower is better): measures similarity of same-segment point embeddings
- **Uniformity** (closer to 0 is better): measures global spread in the embedding space

Best epoch: Alignment ≈ −0.25, Uniformity ≈ −3.0

---

### 4. Multimodal LLM Regressor

The regression head adapts a **frozen Qwen2.5-0.5B** LLM for multimodal time prediction via a cross-modality bridging mechanism.

**Airspace Scenario Construction:**

At each observation time *t*, three trajectory embedding streams are constructed from the airspace snapshot:

| Stream | Description | Embedding |
|---|---|---|
| `focusing_emb` | The flight being predicted | ∈ ℝ²⁵⁶ |
| `active_embs` | Co-flying flights currently in TMA | ∈ ℝᴺˣ²⁵⁶ |
| `prior_embs` | Recently landed flights | ∈ ℝᴹˣ²⁵⁶ |

These are passed through three lightweight **Trajectory Embedding Converters** and concatenated with tokenized text prompts before being fed into the frozen LLM backbone.

**Input Prompts:**

*Flight Plan Prompt* — structured natural language with placeholders:
```
Current time: <OBS_TIME>.
Actual airspace entry time for flight <CALLSIGN> was <ENTRY_TIME>.
This <HAUL_TYPE> flight operated by <AIRLINE> is scheduled to arrive
at <ARR_TIME> on <DATE>.
It originated from <DEP_AIRPORT> (<DEP_ICAO> / <DEP_IATA>),
dep lat: <DEP_LAT>, lon: <DEP_LON>, alt: <DEP_ALT> ft,
and was headed for Istanbul Airport (<DEST_ICAO> / <DEST_IATA>)...
Aircraft type: <AIRCRAFT_TYPE>. Registration: <REG>.
Wake turbulence category: <WTC>. Total route distance: <DISTANCE> km.
```

*Weather Information Prompt* — raw METAR + TAF strings for LTFM at observation time.

**Full Forward Pass:**

```
[Flight Plan Prompt] + [Weather Prompt]
          │
      Tokenizer
          │
  Pretrained Embedding Table
          │
        Concat ← [Trajectory Converters: focusing | active | prior]
          │
   Pre-trained Frozen LLM (Qwen2.5-0.5B)
          │
   Regression Head → Predicted Duration (minutes)
```

---



## Results

Monthly out-of-sample prediction performance on the radar-to-landing duration regression task:

| Month | MAE | MSE | RMSE | R² | Adj. R² | SMAPE (%) |
|---|---|---|---|---|---|---|
| 2025-03 | 2.0206 | 7.3706 | 2.7149 | 0.7934 | 0.7933 | 7.31 |
| 2025-06 | 1.1614 | 4.4891 | 2.1187 | 0.9059 | 0.9059 | 5.62 |
| 2025-07 | 1.2102 | 3.5811 | 1.8924 | 0.9140 | 0.9140 | 6.64 |
| 2025-08 | 1.4517 | 5.0792 | 2.2537 | 0.8864 | 0.8864 | 5.50 |
| 2025-09 | 2.3633 | 12.9002 | 3.5917 | 0.5643 | 0.5641 | 7.21 |
| 2025-10 | 1.3894 | 4.1342 | 2.0333 | 0.9012 | 0.9010 | 6.32 |
| 2025-11 | 2.0692 | 8.5511 | 2.9242 | 0.5474 | 0.5467 | 6.61 |
| 2025-12 | 1.6320 | 7.4413 | 2.7279 | 0.7978 | 0.7977 | 5.57 |
| 2026-01 | 2.2300 | 8.1053 | 2.8470 | 0.7071 | 0.7068 | 6.74 |

**Key findings:**
- Summer months (June–July 2025) yield the strongest performance (R² > 0.90, MAE < 1.25 min), benefiting from higher traffic volume and more consistent approach patterns.
- Autumn/winter months (September, November) show higher variance, likely due to more frequent weather-induced holding events and irregular traffic.
- The model achieves sub-3-minute RMSE across all evaluated months, demonstrating practical viability for ATFM support.

---




**Core dependencies:**

| Package | Role |
|---|---|
| `torch` | ATSCC training & LLM inference |
| `transformers` + `accelerate` | Qwen2.5-0.5B loading & cross-modality adaptation |
| `pandas` + `polars` + `pyarrow` | Data lakehouse pipeline |
| `scikit-learn` | Preprocessing & evaluation metrics |
| `rdp` | Ramer–Douglas–Peucker trajectory segmentation |
| `pymap3d` | ECEF ↔ geodetic coordinate conversion |
| `umap-learn` | Embedding space visualisation |
| `sentencepiece` | Tokenizer support for Qwen |

---

## Acknowledgements

This work draws on and extends the cross-modality adaptation framework for flight delay prediction proposed in:

> Phisannupawong, T., Damanik, J. J., & Choi, H.-L. (2024). *Flight Delay Prediction via Cross-Modality Adaptation of Large Language Models and Aircraft Trajectory Representation.* Department of Aerospace Engineering, KAIST, Republic of Korea.

The self-supervised contrastive learning objective for trajectory segmentation is inspired by Siamese network designs in representation learning literature.

ADS-B data processing uses the [OpenSky Network](https://opensky-network.org/) data format conventions.

---

## Contact

| | |
|---|---|
| **Email** | yasaryigitturan@gmail.com |
| **LinkedIn** | [yaşar-yiğit-turan](https://www.linkedin.com/in/yaşar-yiğit-turan-/) |

---

<p align="center">
  <sub>Department of Statistics · Hacettepe University · Ankara, Turkey</sub>
</p>
