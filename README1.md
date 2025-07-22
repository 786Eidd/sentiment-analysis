# Emotion Classification with LSTM & GloVe Embeddings

This project implements a deep learning-based emotion classification system using TensorFlow/Keras. It takes in short text messages and classifies them into one of six universal emotions: **sadness, joy, love, anger, fear, or surprise**.

---

## Overview

The system is built with:
- **Custom preprocessing** (cleaning, tokenization)
- **TF-IDF** and **sequential word embeddings**
- **Pretrained GloVe vectors**
- **Bidirectional LSTM model**
- **Model saving and prediction interface**

---

## Project Structure
emotion_classifier/
├── data/
│ ├── training.csv
│ ├── validation.csv
│ └── test.csv
│
├── embeddings/
│ └── glove.6B.200d.txt
│
├── models/
│ ├── sentiment_analysis.keras
│ └── tokenizer.pickle
│
├── src/
│ ├── preprocessing.py # Preprocess & clean dataset
│ ├── features.py # TF-IDF & sequence features
│ ├── model.py # LSTM model builder
│ ├── train.py # End-to-end training pipeline
│ └── predict.py # CLI or Streamlit interface
│
├── notebooks/
│ └── sentiment_emotion_classifier.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore


---

##  Emotion Labels

| Label | Emotion   |
|-------|-----------|
| 0     | Sadness   |
| 1     | Joy       |
| 2     | Love      |
| 3     | Anger     |
| 4     | Fear      |
| 5     | Surprise  |

---

## Requirements

Install required packages with:

```bash
pip install -r requirements.txt

Also download GloVe embeddings (glove.6B.200d.txt) from:
https://nlp.stanford.edu/projects/glove/


##  Download Model Files

| File                                                             |  Description                             |
|------------------------------------------------------------------|                                          |
| [`sentiment_analysis.keras`](./models/sentiment_analysis.keras)  | Trained LSTM model with GloVe embeddings |
| [`tokenizer.pickle`](./models/tokenizer.pickle)                  | Tokenizer object for sequence prediction |
