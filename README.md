# Physics-Informed Gaussian Process Regression for Predicting Flow in an Urban Drainage System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code, models, and results for the paper **"Physics-Informed Gaussian Process Regression for Predicting Flow in an Urban Drainage System."** 

## 📖 Overview
Accurate forecasting of urban drainage flows is critical for mitigating environmental pollution and optimizing wastewater treatment. While purely data-driven models are computationally efficient, they often lack physical interpretation and produce unrealistic predictions. 

This repository provides a probabilistic framework using physics-informed Gaussian Process Regression (GPR) to forecast Wastewater Treatment Plant (WWTP) inflows, Combined Sewer Overflows (CSOs), and upstream tank levels. It includes implementations of domain-aware composite kernels, SWMM-derived prior mean functions, physical output constraints, and a novel **stratified Sparse GPR (SGPR)** method designed specifically for event-based time-series.

## 🗂️ Repository Structure

The code is modularized into specific folders to separate the different forecasting objectives discussed in the paper:

* 📁 **`basic_models/`** *(Note: Rename this to match your actual folder name if different)*
  * Contains the shared baseline GPR models, kernel definitions, and utility scripts used across the project.
* 📁 **`WWTP_forecasting/`** 
  * Contains the data preprocessing, model scripts, and results for predicting the WWTP inflow (as discussed in Sections 3.1 & 3.2).
* 📁 **`CSO_forecasting/`** 
  * Contains the scripts and results for direct CSO prediction, Tank Level forecasting, and the novel stratified SGPR models (as discussed in Sections 3.3, 3.4, & 3.5).
* 📄 **`requirements.txt`** 
  * List of Python dependencies required to run the models.

## ⚙️ Installation & Setup

To replicate the results or run the models locally, clone this repository and install the required dependencies. It is recommended to use a virtual environment.
```bash
git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
cd YOUR-REPO-NAME
pip install -r requirements.txt
