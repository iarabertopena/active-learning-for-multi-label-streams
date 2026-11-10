# Active Learning for Multi-Label Data Streams

This repository contains the implementation of an undergraduate research project on **Online Multi-Label Classification (OMLC)**, developed at the Institute of Mathematical and Computer Sciences, University of São Paulo (ICMC-USP), Brazil.

The project is funded by the São Paulo Research Foundation (FAPESP, grant **2026/06454-3**) and supervised by **Prof. Dr. Ricardo Cerri** and **Prof. Dr. Felipe Kenji**.

> **Project status:** Ongoing research. Experimental results are preliminary, and the implemented strategies are still being developed and evaluated.

## Overview

The project investigates Active Learning strategies for multi-label data streams, aiming to reduce annotation costs while maintaining predictive performance.

The experimental framework compares different query strategies under a limited annotation budget, including a proposed approach called **Cluster Hardness Sampling (CHS)** and its combinations with uncertainty-based and committee-based strategies.

### Methodology

- **Classifier:** Binary Relevance (BR) combined with Adaptive Random Forest (ARF), using CapyMOA.
- **Query strategies:** Random Sampling, Uncertainty Sampling, Query-by-Committee (QBC), Cluster Hardness Sampling (CHS), CHS combined with Uncertainty Sampling, and CHS combined with QBC.
- **Reference scenarios:** No Query and Full Supervision.
- **Learning settings:** Online learning and pool-based active learning.
- **Evaluation protocol:** Test-then-train (prequential) evaluation, with the first 30% of observations used for initial training and the remaining 70% processed as a data stream, without shuffling.

### Evaluation Metrics

The experiments consider both predictive performance and annotation efficiency:

- **Hamming Loss**
- **Exact Match**
- **Macro-F1**
- **Query Rate** (percentage of queried instances)
- **Execution Time**

## Datasets

The experimental evaluation includes real-world and synthetic multi-label datasets.

| Type | Datasets |
|---|---|
| Real-world | Flags, CHD-49, Emotions, Scene, Birds, Yeast, Water-quality, Yelp |
| Synthetic | SynHPGrad, SynHPInc |

## Repository Structure

```text
.
├── notebooks/      # Jupyter notebooks
├── results/        # Experimental outputs
├── src/            # Source code
├── README.md
└── requirements.txt
```

## Technologies

- Python
- CapyMOA
- Adaptive Random Forest
- Binary Relevance
- Active Learning
- Multi-Label Classification

## Research Context

This repository is part of an ongoing undergraduate research project in Machine Learning, focusing on Active Learning for Online Multi-Label Classification and annotation-efficient learning from data streams.
