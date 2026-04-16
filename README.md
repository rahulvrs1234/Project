# Diagnosis of a Patient to Detect COVID-19 Using Deep Learning

This repository is a cleaned archival reconstruction of my Bachelor of Technology final-year project on COVID-19 diagnosis using deep learning. The available material was limited to the surviving project description and source-code excerpts, so this repo is designed as a professional source snapshot rather than a guaranteed plug-and-play deployment.

## Project Description

The project focuses on building an intelligent diagnosis-support system for COVID-19 using deep learning, clinical information, and medical imaging. The main idea is to combine structured and unstructured health data into a user-friendly platform that can help accelerate diagnosis, support clinical understanding, and improve prediction workflows.

The project description available from the report emphasizes three major goals:

- improve COVID-19 diagnosis through artificial intelligence and deep learning
- support decision-making using clinical data, public COVID-19 trends, and medical images
- create an integrated platform for physicians, researchers, and users through a web-based workflow

The report highlights the use of Artificial Neural Networks (ANN), Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), and broader AI-assisted methods to analyze symptoms, vitals, risk factors, and imaging data such as chest X-rays and CT scans.

## Project Summary

The recovered material shows a system with multiple connected parts:

- a Django web application for user registration, login, and admin workflows
- COVID-19 current-status analytics using public data sources
- clinical report handling and patient-data processing
- CNN-based training scripts for COVID-19 image classification
- LSTM-based forecasting for case-result analysis
- deep-learning support for detecting infection from imaging input

## Available Source Snapshot

Recovered and reconstructed modules in this repo:

- `src/run_simulator.py` - Django view/controller logic
- `src/train_covid.py` - transfer-learning based CNN training script
- `src/build_covid_model.py` - custom CNN model builder and trainer
- `src/a_util.py` - data-analysis utility helpers for clinical and radiology data
- `src/algorithms/get_current_status.py` - COVID-19 current-status analytics
- `src/algorithms/user_results.py` - LSTM-based forecasting and scoring
- `src/algorithms/get_clinical_reports.py` - placeholder module for unrecovered clinical-report logic

## Technology Stack

- Python
- Django
- TensorFlow / Keras
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- imutils
- requests

## Project Features

- Patient registration and login workflow
- User and admin role separation
- COVID-19 current-status visualization
- Clinical data reporting pipeline
- Chest X-ray and CT-scan analysis workflow
- CNN training for COVID-19 / normal classification
- LSTM-based result prediction from case data

## Repository Structure

```text
Project/
  docs/
    project-overview.md
  src/
    algorithms/
      get_clinical_reports.py
      get_current_status.py
      user_results.py
    a_util.py
    build_covid_model.py
    run_simulator.py
    train_covid.py
  .gitignore
  requirements.txt
  README.md
```

