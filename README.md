# car_t_stimulation
Data and code for analyzing and predicting different CAR-T cell products


# Introduction

This repository contains code to reproduce the computational results for the paper "Enhancing CAR-T cell functionality in a patient-specific manner".

# File Structure

The 'Cytox' folder contains code/data in R for running the cytotoxicity analysis in Figure 2.

The 'bin' folder contains code in python for the computational analysis in Figure 4. The 'utils.py' file contains functions which are implemented in the 'APCms Analysis' Jupyter notebook. The data for these files is stored in 'data', and the results (figures and tuned model parameters) are in the 'results' folder.

# Instructions

Download the repository and ensure that you have the python packages listed in 'requirements.txt'. Open the 'APCms Analysis' notebook and click 'Run All'. In the notebook there are options that one can alter such as repeating parameter tuning, but the default settings and current configuration will produce the results shown in the paper.


For any questions, please email Siddharth Iyer at iyers@mit.edu.
[![DOI](https://zenodo.org/badge/556846863.svg)](https://zenodo.org/badge/latestdoi/556846863)
