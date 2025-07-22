#  Sentiment & Emotion Detection with Deep Learning

This project implements a robust deep learning pipeline for **multi-class emotion classification** from short text inputs using TensorFlow/Keras. It classifies sentences into one of six universal human emotions: `sadness`, `joy`, `love`, `anger`, `fear`, and `surprise`.

Built using **preprocessed GloVe word embeddings**, **LSTM neural networks**, and **custom NLP preprocessing**, the model achieves accurate, interpretable predictions suitable for chatbots, social media monitoring, or real-time sentiment feedback systems.

---

##  Emotion Classes

| Label | Emotion   | Example Text                      |
|-------|-----------|-----------------------------------|
| 0     | Sadness   | "I feel heartbroken."             |
| 1     | Joy       | "Life is beautiful!"              |
| 2     | Love      | "I truly adore my family."        |
| 3     | Anger     | "This is so frustrating!"         |
| 4     | Fear      | "I'm worried about tomorrow."     |
| 5     | Surprise  | "Wow, that was unexpected!"       |

---

## Project Structure

```
emotion-classifier/
├── data/
├── embedding/
├── models/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
```

---

## Features

-  Data cleaning & tokenization using `nltk` & `re`
-  Feature extraction using `TF-IDF` and `Keras Tokenizer`
-  GloVe (Global Vectors) embedding integration
-  LSTM-based emotion classification model
-  One-hot encoding of emotion labels
-  GPU memory limiting (optional TensorFlow optimization)
-  Model serialization to `.keras` and tokenizer to `.pickle`
-  Ready-to-integrate prediction function with softmax outputs

---

##  Getting Started

### 🔧 Installation

```bash
pip install -r requirements.txt
```

###  Place your files like this:

```
data/
  training.csv
  validation.csv
  test.csv

embedding/
  glove.6B.200d.txt
```

Download GloVe embeddings from:
 https://nlp.stanford.edu/projects/glove/

---

### Training the Model

```bash
python src/train.py
```

- Model is saved as: `models/sentiment_analysis.keras`
- Tokenizer is saved as: `models/tokenizer.pickle`

---

### Predicting Emotion

```bash
python src/predict.py
```

In Python:

```python
from src.predict import predict_sentiment
predict_sentiment("I feel amazing today!")  # returns 'joy'
```

---

## Model Architecture

- Embedding layer (preloaded GloVe vectors)
- LSTM (512) → LSTM (256)
- Dense (128 + dropout) → Dense (64)
- Output: Softmax layer with 6 classes

---

## Sample Results

```text
Epoch 20/20
val_accuracy: 0.87
train_accuracy: 0.91

Example prediction:
Input: "I'm afraid of the outcome"
Output: fear (label 4)
```


---

## Acknowledgements

- TensorFlow/Keras
- GloVe Embeddings (Stanford NLP)
- NLTK
- Inspired by Paul Ekman's Six Basic Emotions Framework

---

## Author

**Eid M.**  
AI/ML Engineer | NLP Enthusiast  
[GitHub](https://github.com/786Eidd) • [LinkedIn](https://www.linkedin.com/in/eidmo/)
