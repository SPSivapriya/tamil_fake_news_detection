# Tamil Fake News Detection

This project focuses on detecting fake and real Tamil news headlines using a machine learning pipeline built in Python. It was developed as a UG Creative and Innovative Project (CIP) at Anna University and is implemented as a single Jupyter Notebook workflow.

The notebook follows a full NLP pipeline for text classification, starting from data loading and exploration, moving through preprocessing, feature extraction, model design, training, and final prediction.

## Project Overview

The goal of this project is to classify Tamil news headlines as either:

- Fake (0)
- Real (1)

The system is designed for short-form news text and uses a sequence-based deep learning approach with preprocessing adapted to Tamil text and mixed-language headlines.

## What is included

The notebook covers the following stages:

1. Data collection and dataset inspection
2. Analysis of label distribution and sample headlines
3. Tamil-specific text preprocessing
4. Feature extraction using TF-IDF, Word2Vec, and FastText representations
5. Model adaptation for low-resource Tamil data
6. Training a deep learning model for binary classification
7. Evaluation with confusion matrix and classification metrics
8. Inference for new headline prediction

## Dataset

The project expects a CSV file named `Tamil-News-Headlines.csv`.

The dataset is a Tamil news headlines dataset containing text and a binary authenticity label:

- `News`: the headline text
- `Authenticity`: 0 for fake and 1 for real

The notebook reports approximately 5,226 samples, with a near-balanced distribution across fake and real labels.

## Project Structure

At the moment, the repository contains a single notebook:

- `100_implementation.ipynb` - complete implementation of the pipeline

## Setup

To run the project locally, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn gensim tensorflow
```

It is recommended to use a Jupyter Notebook environment or JupyterLab so the notebook cells can be executed in order.

## Running the Notebook

1. Open `100_implementation.ipynb` in Jupyter.
2. Run the first code cell to install the dependencies if needed.
3. Upload or place the dataset file `Tamil-News-Headlines.csv` in the same working directory.
4. Execute the notebook cells in sequence.
5. Use the final prediction cell to enter a Tamil, English, or mixed-language headline and get a fake/real result.

## Model and Methodology

The notebook uses a practical machine learning workflow for short Tamil text:

- Text cleaning and normalization
- Removal of URLs, special characters, and noise
- Tamil-aware preprocessing with stopword removal and light stemming
- Train-validation-test split before feature extraction to avoid data leakage
- TF-IDF and embedding-based feature extraction
- LSTM-based model adaptation for classification
- Early stopping and evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix

## Notes

This repository is focused on educational and research-oriented implementation rather than a production-ready package. The notebook is designed to be easy to follow and to show the complete workflow of a fake news detection system for Tamil text.

## License

This project does not include a formal license file. If you plan to reuse or extend the code, it is best to confirm the licensing terms with the project owner before distributing or publishing it.
