import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

labelled_reviews = pd.read_csv(r"C:\Users\malup\Desktop\Projects\NLP\Fake reviews detection\Datasets\fake reviews dataset.csv")
labelled_reviews['length'] = labelled_reviews['text_'].apply(len)

unlabelled_reviews = pd.read_csv(r"C:\Users\malup\Desktop\Projects\NLP\Fake reviews detection\Datasets\train.csv")[0:40000]
unlabelled_reviews.columns = ['label', 'title', 'text_']

# Split the labeled data into training and validation sets
train_data, val_data = train_test_split(labelled_reviews, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_vectorized = vectorizer.fit_transform(train_data['text_'])
X_val_vectorized = vectorizer.transform(val_data['text_'])
X_unlabeled_vectorized = vectorizer.transform(unlabelled_reviews['text_'])

# Train the initial model on the labeled data
model = LogisticRegression()
model.fit(X_train_vectorized, train_data['label'])

# Evaluate initial model performance
initial_predictions = model.predict(X_val_vectorized)
initial_accuracy = accuracy_score(val_data['label'], initial_predictions)
print("Initial Accuracy:", initial_accuracy)

# Pseudo-labeling: Predict labels for the unlabeled data
pseudo_labels = model.predict(X_unlabeled_vectorized)

# Add pseudo-labeled data to the training set
pseudo_labeled_data = unlabelled_reviews.copy()
pseudo_labeled_data['label'] = pseudo_labels
new_train_data = pd.concat([train_data, pseudo_labeled_data])

# Retrain the model on the combined dataset
X_new_train_vectorized = vectorizer.transform(new_train_data['text_'])
model.fit(X_new_train_vectorized, new_train_data['label'])

# Evaluate updated model performance
updated_predictions = model.predict(X_val_vectorized)
updated_accuracy = accuracy_score(val_data['label'], updated_predictions)
print("Updated Accuracy:", updated_accuracy)

# Save the trained model to a file
joblib.dump(model, 'logistic_regression_model.pkl')
