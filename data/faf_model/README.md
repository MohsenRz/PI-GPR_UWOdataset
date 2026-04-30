# Fehraltorf/UWO case study

This repository contains SWMM model file(s) and adjacent data files for the Fehraltorf/UWO case study.

## Model File (root folder) 
<b>faf.inp</b>
EPA SWMM file Fehraltorf | validated model | see detailed description in SI, Chapter 7

## Input data (root folder)
- measured rainfall from raingage "r02 - school" -> r02_mm_utc0_1min_feb16_apr20.dat
- measured transfer flows from Rumlikon -> f02_rum_lps_feb16_apr20.dat
- measured transfer flows from Russikon -> f03_rus_lps_mar16_apr20.dat
- OPTIONAL: measured rainfall from Kloten, Zurich (airport, MeteoSwiss) -> klo_mm_utc0_10min_jan1981_aug2017.dat

## Background map (root folder)
<b>Backdrop_Fehraltorf_Swissimage.bmp</b>

## Reference data
<b>_ref_data</b>
This folder contains validated WWTP inflow observations used for calibration and validation exercises. The readme file explains observed flows data origin. 
- link_2_f00_lps_utc0_5min_apr16_sep19.dat
- readme_link_2.txt

## Model files for neighbouring catchments
<b>rum_model</b>
This folder contains the EPA SWMM model file for Rumlikon only (untested). The readme file indicates information about the model and its modification. 

<b>rus_model</b>
This folder contains the EPA SWMM model file for Russikon only (untested). The readme file indicates information about the model and its modification. 

<b>faf_rus_model</b>
This folder contains the EPA SWMM model file for Fehraltorf and Russikon as conjoint network (untested).

## To consider
When using the EPA SWMM model files, be aware that you may need to change the pathnames pointing to the corresponding *.dat files.