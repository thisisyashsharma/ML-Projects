# Spotify Genre Segmentation — Corizo Internship

[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()

## Project summary

**Goal:** build a reproducible, memory-efficient ML pipeline to segment Spotify tracks into clusters using audio features (danceability, energy, valence, tempo, loudness, etc.). The output is a `track_id → cluster_label` mapping and a saved pipeline artifact for inference.

This repo contains the code, configuration and small utilities used during the Corizo Private Limited internship (Mar 2024 – May 2024).

## What we built (high level)

- Chunked data ingestion and dtype downcasting to reduce memory footprint.
- Feature preprocessing (scaling + optional PCA).
- Scalable clustering using `MiniBatchKMeans`.
- A simple CLI for end-to-end runs and a saved joblib artifact (`models/pipeline.joblib`).
- Lightweight EDA notebooks (not included in this branch) and supporting scripts.

## Quickstart (Windows PowerShell & cross-platform)

> Clone the repository (example):
```bash
git clone https://github.com/thisisyashsharma/ML-Projects.git
cd ML-Projects
