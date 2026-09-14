# Physics-Constrained Gaussian Process Regression for Predicting Flow in an Urban Drainage System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Preprint](https://img.shields.io/badge/Preprint-EarthArXiv-orange.svg)](https://doi.org/10.31224/7256)

This repository contains the official code, models, and results for the paper **"Physics-Informed Gaussian Process Regression for Predicting Flow in an Urban Drainage System."** 

## 📖 Overview
Accurate forecasting of urban drainage flows is critical for mitigating environmental pollution and optimizing wastewater treatment. While purely data-driven models are computationally efficient, they often lack physical interpretation and produce unrealistic predictions. 

This repository provides a probabilistic framework using Gaussian Process Regression (GPR) to forecast Wastewater Treatment Plant (WWTP) inflows, Combined Sewer Overflows (CSOs), and upstream tank levels. It includes implementations of domain-aware composite kernels, SWMM-derived prior mean functions, physical output constraints, and a novel **stratified Sparse GPR (SGPR)** method designed specifically for event-based time-series.

## 🗂️ Repository Structure

The code is modularised into specific folders to separate the different forecasting objectives discussed in the paper:

* 📁 **`data/`** 
  * Contains the SWMM models and reference data provided by the UWO dataset.
* 📁 **`WWTP_flow_prediction/`** 
  * Contains the model scripts and results for predicting the WWTP inflow (as discussed in Sections 3.1 & 3.2).
* 📁 **`CSO_prediction/`** 
  * Contains the scripts and results for direct CSO prediction, Tank Level forecasting, and the novel stratified SGPR models (as discussed in Sections 3.3, 3.4, & 3.5).
* 📄 **`environment.yml`** 
  * Conda environment file with all Python dependencies required to run the models.
* 📄 **`calling_data.py`**
  * Script for loading and processing raw data from the external data source.

## 🌿 Branches
For full datasets, extended results, and additional scripts, see the dedicated branches:
- [**`WWTP_inflow_forecasting`**](https://github.com/MohsenRz/PI-GPR_UWOdataset/tree/WWTP_inflow_forecasting) — Full data and results for WWTP inflow prediction. 
- [**`CSO_prediction`**](https://github.com/MohsenRz/PI-GPR_UWOdataset/tree/CSO_forecasting) — Full data and results for CSO and tank level forecasting.

## 📦 Data
Raw data files are not included in this repository due to size constraints. The original datasets can be accessed from the UWO Dataset available at: [Eawag Open Data - UWO Field Observations](https://opendata.eawag.ch/dataset/uwo_field-observations_2019_to_2021)

Once downloaded, please use the provided `calling_data.py` script to process the raw data and convert them into `.pkl` (pickle) files for efficient model training.

## ⚙️ Installation & Setup

Clone this repository and recreate the conda environment:
```bash
git clone [https://github.com/MohsenRz/PI-GPR_UWOdataset.git](https://github.com/MohsenRz/PI-GPR_UWOdataset.git)
cd PI-GPR_UWOdataset
conda env create -f environment.yml
conda activate myenv1
```
## 📝 Citation
If you find this code, data, or framework useful in your research, please consider citing our preprint:

```bibtex
@article{rezaee2026physics,
  title={Physics-Constrained Gaussian Process Regression for Predicting Flow in an Urban Drainage System},
  author={Rezaee, Mohsen and Melville-Shreeve, Peter and Rappel, Hussein},
  journal={EarthArXiv},
  year={2026},
  doi={10.31224/7256},
  url={[https://doi.org/10.31224/7256](https://doi.org/10.31224/7256)}
}
```