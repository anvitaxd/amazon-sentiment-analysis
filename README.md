# amazon-sentiment-analysis
Sentiment Analysis on Amazon Product Reviews using NLP

Overview : This project uses Natural Language Processing (NLP) and Machine Learning to analyze Amazon product reviews and classify them as Positive or Negative. It demonstrates text preprocessing, feature extraction, model training, and evaluation on the Amazon Polarity Dataset.

Methodology & Working : 
1) Data Loading - Amazon Polarity Dataset is loaded with review text and sentiment labels (0 = Negative, 1 = Positive).
2) Data Preprocessing -
~Lowercasing and text normalization
~Tokenization and stopword removal
~Stemming/Lemmatization
~Conversion to numerical features using TF-IDF or Bag-of-Words
4) Model Building - 
~Split dataset into training and test sets.
~Train a machine learning model (e.g., Logistic Regression / Naive Bayes).
~Evaluate using accuracy, precision, recall, and F1-score.
5) Evaluation - 
~Generate classification report and confusion matrix.
~Analyze performance and potential improvements.
6) Visualization - 
~Display class distribution and feature importance.
~Generate word clouds to visualize frequent terms in positive and negative reviews.

Model Performance :

1.Accuracy: 85.68%

2.Precision: Negative = 0.85 | Positive = 0.86

3.Recall: Negative = 0.85 | Positive = 0.86

4.F1-Score: Negative = 0.85 | Positive = 0.86


Visualizations:

✅ Class Distribution Plot

<img width="756" height="583" alt="image" src="https://github.com/user-attachments/assets/1bfebf25-7c2c-4a78-8008-7155a37a1957" />



✅ Word Cloud of Positive vs. Negative Reviews

<img width="577" height="484" alt="image" src="https://github.com/user-attachments/assets/1ec3b54c-aaec-44b0-bc8e-2f94211a4625" />

<img width="587" height="523" alt="image" src="https://github.com/user-attachments/assets/5fb77c9c-e785-42bb-a0a6-3856345d5fcb" />




✅ Confusion Matrix Heatmap

<img width="705" height="609" alt="image" src="https://github.com/user-attachments/assets/3686a101-d03c-4efe-b04b-da0698a1c9b6" />



Dataset :

1) Amazon Polarity Dataset

2) Millions of Amazon product reviews labeled as Positive/Negative.

3) Kaggle Dataset Link - https://www.kaggle.com/bittlingmayer/amazonreviews




Installation :

1) Clone the repository:
git clone https://github.com/yourusername/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis

2) Install dependencies:
pip install -r requirements.txt


Usage:

Run the Jupyter Notebook:
jupyter notebook "Amazon Review Sentiment Analysis.ipynb"


Requirements:

1) Python 3.8+
2) Numpy
3) Pandas
4) Matplotlib
5) NLTK
6) Scikit-learn


Future Enhancements :

1) Use deep learning models (LSTM, BERT) for better accuracy.

2) Deploy the model as a web app for real-time review analysis.
   
3) Perform hyperparameter tuning for improved performance.


Contributing :

Contributions are welcome! Please open an issue or submit a pull request.
