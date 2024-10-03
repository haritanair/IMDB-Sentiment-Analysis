# Sentiment Analysis on IMDB Reviews

## Overview

This project aims to classify sentiments (positive or negative) from IMDB movie reviews using traditional machine learning models and deep learning approaches. We compare the performance of several models on the dataset and interpret their results.

## Dataset

The dataset used in this project can be found on Kaggle: [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Dataset**: IMDB movie reviews, containing 50,000 reviews labeled as positive or negative.
- **Target Variable**: Sentiment (positive/negative).

### Exploratory Data Analysis (EDA)
- **Sentiment Visualization**: Distribution of positive and negative sentiments in the dataset.
- **Word Cloud**: A visual representation of the most frequent words for each sentiment.
- **Common Words**: Identification of common words in positive and negative reviews.
- **Review Length Analysis**: Study of the distribution of review lengths to understand text variability.

## Data Preprocessing

### For Traditional Models

#### Steps:
1. **Handle Missing Data**: Removed any rows with missing values.
2. **Remove Duplicates**: Removed duplicate reviews.
3. **Expand Contractions**: Expanded words like "don't" to "do not" using a contraction mapping.
4. **Handle Emojis**: Converted emojis to text descriptions (e.g., 😊 → :smiley:).
5. **Normalize Repeated Characters**: Reduced repeated characters (e.g., "looove" → "loove").
6. **Clean the Text**:
    - Lowercased all text.
    - Removed HTML tags.
    - Removed punctuation and special characters.
    - Tokenized the text.
    - Removed stopwords.
    - Handled negations by appending "not_" to words following "not" (e.g., "not good" → "not_good").
    - Performed lemmatization with POS tagging.
7. **Remove Rare Words**: Removed words that appeared fewer than five times across the dataset.
8. **Prepare Data**: Split the cleaned dataset into training and test sets (80:20 split).

#### Tokenization for Traditional Models:
- Tokenized the text and applied the necessary cleaning steps before passing it to traditional models.

### For Deep Learning Models

#### Steps:
1. **Label Mapping**: Mapped string labels (`positive`, `negative`) to integers (1 for positive, 0 for negative).
2. **Clean the Text**:
   - Lowercased the reviews.
   - Removed punctuation.
3. **Tokenization**:
   - Used a tokenizer with a maximum of 5000 words based on word frequency.
   - Reduced the number of tokens due to computational limitations (initially 5000, reduced to 500).
4. **Text to Sequences**: Converted text to sequences of integers.
5. **Padding**: Padded sequences to ensure uniform input size (maximum sequence length of 100).
6. **Split Data**: Split into training and testing sets with an 80:20 ratio.

## Models Used

1. **Logistic Regression**
2. **Random Forest**
3. **Naive Bayes**
4. **Support Vector Machine (SVM)**: Initially faced RAM issues in Colab, switched to `LinearSVC` which executed successfully.
5. **LSTM**
6. **GRU**
7. **CNN**
8. **Ensemble Model**: Combined results from various models for improved accuracy.

### Tokenization for Deep Learning Models:
Used a tokenizer for deep learning models with word sequences converted into integer sequences. Maximum word count reduced due to RAM constraints in Colab.

## Model Performance and Interpretation

| Model               | Accuracy | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) | ROC AUC Score |
|---------------------|----------|---------------|------------|--------------|---------------|------------|--------------|---------------|
| Logistic Regression  | 84.29%   | 0.84          | 0.83       | 0.83         | 0.84          | 0.85       | 0.84         | 0.918         |
| Random Forest        | 80.61%   | 0.81          | 0.79       | 0.80         | 0.80          | 0.82       | 0.81         | 0.890         |
| Naive Bayes          | 82.10%   | 0.83          | 0.80       | 0.82         | 0.81          | 0.84       | 0.83         | 0.900         |
| SVM                  | 84.00%   | 0.85          | 0.82       | 0.83         | 0.83          | 0.86       | 0.84         | -             |
| LSTM                 | 84.00%   | 0.84          | 0.84       | 0.84         | 0.85          | 0.84       | 0.84         | 0.9207        |
| GRU                  | 84.00%   | 0.86          | 0.81       | 0.83         | 0.82          | 0.87       | 0.85         | 0.9232        |
| CNN                  | 85.00%   | 0.84          | 0.87       | 0.85         | 0.86          | 0.83       | 0.85         | 0.9328        |
| Ensemble Model       | 86.00%   | 0.86          | 0.86       | 0.86         | 0.86          | 0.87       | 0.86         | 0.9380        |

### Interpretation:
- **Accuracy**: Measures the proportion of correctly classified reviews.
- **Precision**: Indicates the proportion of correctly predicted positive/negative reviews out of all predicted positives/negatives.
- **Recall**: Indicates how well the model identifies all actual positive/negative reviews.
- **F1-Score**: Harmonic mean of precision and recall. It gives a balance between precision and recall.
- **ROC AUC Score**: AUC-ROC score reflects the model’s ability to differentiate between positive and negative classes. Higher values indicate better performance.

### Observations:
- The **Ensemble Model** showed the best overall performance with the highest accuracy (86%) and ROC AUC score (0.9380).
- **CNN** performed better than traditional models like Random Forest and Naive Bayes.
- **Logistic Regression** and **SVM** provided strong results among traditional methods but fell behind deep learning approaches.
- **Deep Learning Models** (LSTM, GRU, CNN) provided competitive performance, with CNN slightly outperforming the others.

## Challenges and Limitations

- **Tokenization**: Initially attempted tokenization for 5000 words but had to reduce it to 500 due to RAM and computation constraints.
- **Cross-Validation**: Performed cross-validation for only the Logistic Regression model due to resource limitations.
- **BERT & DistilBERT**: Tried using BERT and DistilBERT, but Colab could not execute them due to RAM limitations. This resulted in switching to lighter models.

## Required Packages

Install the following packages to reproduce the project:

```bash
pip install pandas numpy scikit-learn beautifulsoup4 nltk keras tensorflow emoji contractions
