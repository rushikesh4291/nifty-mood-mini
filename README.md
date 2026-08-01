# NIFTY Market Mood Tracker

A lightweight, end-to-end data pipeline and scoring engine designed to compute a daily **Market Mood Score (0-100)** for the NIFTY index. 

This project demonstrates the practical application of data engineering and quantitative analysis by merging institutional cash flows with technical market indicators to synthesize a single, actionable sentiment score. 

## 🚀 Key Features

* **Automated Data Ingestion:** Fetches daily OHLCV data for NIFTY and BANKNIFTY via `yfinance` and merges it with FII/DII net cash flow data.
* **Quantitative Indicators:** Computes RSI (14), Anchored VWAP (monthly & weekly), 20-day volume z-scores, and rolling correlation metrics.
* **Rule-Based Scoring Engine:** Evaluates market conditions against configurable thresholds to generate a weighted mood score.
* **Dashboard-Ready Output:** Automatically exports processed data to a structured CSV format, optimized for immediate visualization.

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** `pandas`, `yfinance`
* **Configuration:** YAML
* **Visualization:** Power BI, Google Sheets

## 📂 Project Architecture

* `data/fii_dii_sample.csv` — Raw FII/DII net cash flows (INR Crore).
* `src/etl.py` — Handles OHLCV data extraction and merging pipelines.
* `src/indicators.py` — Computes technical indicators and rolling statistics.
* `src/alerts.py` — The core rule engine generating boolean flags and weighted scores.
* `src/mood_score.py` — The main pipeline entry point for generating the daily dataset.
* `rules.yaml` — Configurable thresholds and weightings for the scoring model.
* `docs/method_note.md` — Detailed breakdown of the quantitative methodology.
* `dashboards/powerbi_model_spec.md` — Wire-up guide for integrating the output with Power BI.

## ⚙️ Quickstart Guide

**1. Clone and Setup Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
