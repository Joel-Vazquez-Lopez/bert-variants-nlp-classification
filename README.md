# Sentiment Classification with Transformer Models

## Overview

In this project, I implement a sentiment polarity classification system using transformer-based language models within a PyTorch pipeline. The goal is to explore how pre-trained models such as BERT, DistilBERT, and Multilingual BERT perform in practice, and to compare them with a classical machine learning baseline (TF-IDF + Logistic Regression).

The experiments are conducted on the IMDB Large Movie Review Dataset and extended with additional analyses on generalization and multilingual representations. The project focuses not only on model performance, but also on understanding how architectural choices, training strategies, and data conditions affect results.

---

## Objective

The main objective of this project is to evaluate the effectiveness of transformer-based models for sentiment classification and to compare them with traditional approaches.

More specifically, the project explores:

- The performance of pre-trained transformer models (BERT, DistilBERT, Multilingual BERT)
- The impact of model architecture (single-layer vs two-layer classifier)
- The comparison between transformer models and classical ML baselines
- The effect of training conditions (dataset size, epochs)
- The generalization ability of models across domains (IMDB → Amazon reviews)

---

## Approach

I designed a complete sentiment classification pipeline that includes:

### Data Handling
- IMDB Large Movie Review Dataset (50,000 samples)
- Subsampling strategy for efficient experimentation:
  - 10,000 training samples
  - 2,000 test samples
- Full dataset experiments performed locally for final evaluation

### Baselines
- **Random baseline** (uniform predictions)
- **TF-IDF + Logistic Regression**
  - Strong classical baseline for comparison

### Transformer Models
- BERT (`bert-base-uncased`)
- DistilBERT (`distilbert-base-uncased`)
- Multilingual BERT (`bert-base-multilingual-cased`)

### Model Design
- Custom PyTorch implementation (no pre-built pipelines)
- Transformer encoder + classification head
- Two variants:
  - Single linear layer
  - Two-layer classifier (MLP-style)

### Training Setup
- PyTorch `Dataset` and `DataLoader`
- Batch training (batch size = 8)
- Optimizer:
  - AdamW (full experiments)
- Loss function:
  - CrossEntropyLoss
- Training experiments:
  - 2 epochs (baseline comparison)
  - 3 epochs (extended analysis)

### Additional Experiments
- **Generalization experiment**
  - Train on IMDB
  - Test on Amazon reviews
- **Multilingual representation exploration**

---

## Results

### Full Dataset Results

| Model                         | Test Accuracy (%) |
|------------------------------|------------------|
| Baseline                     | 51.00            |
| BERT                         | 88.26            |
| BERT (Two Layers)            | 88.10            |
| DistilBERT                   | 87.25            |
| Multilingual BERT            | 84.38            |
| TF-IDF + Logistic Regression | 85.80            |

---

### Key Observations

- Transformer models significantly outperform the random baseline.
- **BERT achieves the best performance**, with DistilBERT performing very similarly despite being lighter.
- The **two-layer classifier does not improve performance**, suggesting that additional complexity is not necessary for this task.
- **Multilingual BERT performs worse**, likely due to its broader but less specialized representation.
- The **TF-IDF baseline remains highly competitive**, highlighting that classical methods can still perform strongly on structured tasks like sentiment classification.

---

### Training Dynamics

- Increasing the number of epochs improves training accuracy but introduces **mild overfitting**.
- All models show:
  - decreasing loss
  - increasing accuracy during training
- The gap between training and test accuracy suggests the need for further regularization or hyperparameter tuning.

---

### Generalization

- BERT trained on IMDB achieves strong performance on Amazon reviews (~85%)
- This suggests that transformer models capture **domain-independent sentiment features**
- However, results should be interpreted cautiously due to dataset size

---

## Repository Structure
```
.
├── Sentiment_Classification_with_Contextualized_Models.ipynb
├── uncapped.py
├── README.md
```

- `Sentiment_Classification_with_Contextualized_Models.ipynb`  
  Full implementation including data processing, training, evaluation, and analysis.

- `uncapped.py`  
  Script used for full-scale experiments on the complete dataset (50,000 samples).

---

## Future Work

Possible extensions of this project include:

- Hyperparameter tuning (learning rate, batch size, dropout)
- Regularization strategies to reduce overfitting
- Fine-tuning deeper transformer layers
- Evaluation on additional out-of-domain datasets
- Incorporating more advanced transformer architectures


---
