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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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
model = SVC()
model.fit(X_train_vectorized, train_data['label'])

# Evaluate initial model performance
initial_predictions = model.predict(X_val_vectorized)
initial_accuracy = accuracy_score(val_data['label'], initial_predictions)
initial_precision = precision_score(val_data['label'], initial_predictions, average='weighted')
initial_recall = recall_score(val_data['label'], initial_predictions, average='weighted')
initial_f1 = f1_score(val_data['label'], initial_predictions, average='weighted')

print("Initial Performance:")
print("Accuracy:", initial_accuracy)
print("Precision:", initial_precision)
print("Recall:", initial_recall)
print("F1-score:", initial_f1)

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
updated_precision = precision_score(val_data['label'], updated_predictions, average='weighted')
updated_recall = recall_score(val_data['label'], updated_predictions, average='weighted')
updated_f1 = f1_score(val_data['label'], updated_predictions, average='weighted')

print("\nUpdated Performance:")
print("Accuracy:", updated_accuracy)
print("Precision:", updated_precision)
print("Recall:", updated_recall)
print("F1-score:", updated_f1)

# Additional classification report
print("\nClassification Report:")
print(classification_report(val_data['label'], updated_predictions))

# Save the trained model to a file
joblib.dump(model, 'svm_model.pkl')