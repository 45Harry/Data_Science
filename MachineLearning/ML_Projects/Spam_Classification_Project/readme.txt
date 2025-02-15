# Spam Detection Model - README

## Overview
This project aims to detect spam messages using natural language processing (NLP) and machine learning techniques. The model is trained on a dataset of SMS messages and classifies them as either spam or ham (non-spam). The implementation includes data preprocessing, exploratory data analysis (EDA), feature extraction, and model training using various machine learning algorithms.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python (>=3.7)
- pip (Python package manager)
- virtualenv (optional but recommended for dependency management)

## Setting Up a Virtual Environment (Recommended)
It is advisable to use a virtual environment to manage dependencies efficiently.

### Create a Virtual Environment
```sh
python -m venv spam_detection_env
```

### Activate the Virtual Environment
- **Windows:**
  ```sh
  spam_detection_env\Scripts\activate
  ```
- **Mac/Linux:**
  ```sh
  source spam_detection_env/bin/activate
  ```

## Install Dependencies
Once the virtual environment is activated, install the required libraries:
```sh
pip install -r requirements.txt
```

### Required Libraries
The `requirements.txt` file contains:
```
pandas
numpy
seaborn
matplotlib
scikit-learn
nltk
wordcloud
xgboost
```

## Running the Project
After setting up the environment and installing dependencies, execute the following command to run the script:
```sh
python spam_detection.py
```

## Model Training & Evaluation
The script performs the following steps:
1. Loads and cleans the dataset.
2. Preprocesses text data (tokenization, stemming, stopword removal, etc.).
3. Extracts features using TF-IDF vectorization.
4. Splits the dataset into training and testing sets.
5. Trains multiple machine learning models and evaluates performance.
6. Saves the best-performing model using pickle.

## Model and Vectorizer Saving
The trained model and vectorizer are saved for later use:
```sh
vectorizer.pkl
model.pkl
```

## Predicting with the Trained Model
To use the saved model for prediction:
```python
import pickle
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
```

## License
This project is open-source and available for use under the MIT License.

