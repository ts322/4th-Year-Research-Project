# 4th-Year Research Project

> **Working title:** 2D Reactor simulation + VAE-based geometry/data tooling
> **Author:**Thomas Slade & Marco Barbacci Schettino 
> **License:** MIT

---

## Overview

This repository contains code and assets for a fourth‑year research project exploring:

* A **2D reactor** workflow (mesh generation, simulation helpers, post‑processing)
* A **Variational Autoencoder (VAE)** component for geometry generation 

The repo is organized so you can iterate on simulation results while experimenting with generative models.

---

## Repository structure

```
4th-Year-Research-Project/
├─ 2D_Reactor/               # Core reactor-related code & scripts
│  ├─ generate_mesh_*        # Mesh generation utilities
│  ├─ Results/               # **Ignored** outputs from runs (plots, CSVs, 
├─ VAE/                      # Variational Autoencoder training/inference
│  ├─ __init__.py
│  ├─ vae_data_*.py          # Data handling utilities for vae training
│  └─ ...
├─ Data.sh                   # Running the base sim in swakless-1 (bash)
├─ DataMacos*.sh             # Macos compatible data.sh (bash)
├─ Shape_Generation*         # Shape/geometry utilities (scripts)
├─ LICENSE                   # MIT license
└─ README.md                 # This file
```

---

## Quick start

### 1) Clone

```bash
git clone https://github.com/ts322/4th-Year-Research-Project.git
cd 4th-Year-Research-Project
```

### 2) Python environment (if using the VAE / Python tools)

```bash

```

---
**TODO:** add a section on how to install docker for macos pipeline and one for wsl integration for windows 



## Usage

### Mesh / Reactor workflow

```bash
cd 2D_Reactors #if in the 4TH-YEAR-RESEARCH-PROJECT driectory 
```

### VAE training / inference

```bash
#Generate data for the VAE through: 
  python VAE/vae_data_prep.py \
    --out-root ~/ResearchProject/4th-Year-Research-Project/2D_Reactor/VAE/vae_data_prep.py \
    --n-samples 100 \
    --docker-image opencfd/openfoam-default:2506

# Train the VAE in:
python VAE/train.py --data data/ --epochs 100 --latent-dim 64 --out 2D_Reactor/Results/vae_runs/run_001


```
---

## Reproducing experiments

**TODO:** Add a file with configurations for vae training 

---

## Project status

* **Stage:** in active development 

## Contribution 

**Pull requests and issues are welcomed**

## License

This project is licensed under the [MIT License](LICENSE).

