# AI-Powered Parkinson’s Disease Detection System

This project is an interactive Streamlit-based machine learning application designed to detect Parkinson’s Disease using biomedical voice measurement data. The system trains and evaluates multiple machine learning and deep learning models and provides real-time predictions along with detailed analytical visualizations.

Live Application:  
https://my-colab-app-kgowg3c7sbjwhehtcydp8k.streamlit.app/

---

## Project Overview

Parkinson’s Disease is a progressive neurological disorder that affects speech and motor functions. Early detection can support timely medical intervention. This project applies machine learning and deep learning techniques to classify Parkinson’s Disease based on voice-related features.

The application enables users to train multiple models, compare their performance, analyze dataset patterns, and generate predictions using different input methods.

---

## Dataset

- Source: UCI Machine Learning Repository  
- Dataset Name: Parkinson’s Disease Voice Measurements  
- Total Samples: 195  
- Number of Features: 22  
- Target Variable:
  - 0 – Healthy
  - 1 – Parkinson’s Disease

---

## Models Implemented

The system supports training and evaluation of the following models:

- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Multi-Layer Perceptron (Neural Network)  
- Deep Neural Network (TensorFlow, optional)

Each model is trained on standardized data and evaluated using accuracy and confusion matrices.

---

## Application Features

- Interactive model selection and training  
- Dynamic comparison of model accuracy  
- Dataset analysis and visualization  
- Correlation analysis of key features  
- Multiple prediction modes:
  - Manual feature input
  - Random sample selection
  - CSV upload
- Ensemble prediction based on multiple models

---

## Technologies Used

- Python  
- Streamlit  
- Scikit-learn  
- TensorFlow and Keras (optional)  
- Pandas and NumPy  
- Matplotlib and Seaborn  
- Plotly  

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   cd <repository-folder>
Install dependencies:

pip install -r requirements.txt
Run the Streamlit application:

streamlit run app.py
Project Structure
app.py – Main Streamlit application

requirements.txt – Project dependencies

README.md – Project documentation

Results
The system achieves high classification accuracy across multiple models, with neural network-based approaches typically performing best. The application allows users to visually compare results and select the most effective model for predictions.
