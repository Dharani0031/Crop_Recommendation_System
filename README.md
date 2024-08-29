# Crop Recommendation System

## Overview

This Crop Recommendation System is a machine learning application developed using Python's Streamlit library. It allows users to input soil and environmental parameters and receive recommendations on which crop to grow based on their inputs.

## Features

- Input the following parameters:
  - Nitrogen content (N)
  - Phosphorous content (P)
  - Potassium content (K)
  - Temperature (in °C)
  - Humidity (in %)
  - pH value of the soil
  - Rainfall (in mm)

- The system uses a K-Nearest Neighbors (KNN) classifier to predict the most suitable crop.
- Visualizations of the temperature and pH distributions with user inputs highlighted.

## Requirements

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

You can install the required libraries using `pip` command.

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
