# Fake-Reviews-Detection
NLP Python Project that identifies whether reviews left for products are AI-generated or not.

Methods:
1. Semi-supervised learning
Steps:
1. Split your Data:
Divided the dataset into two parts: a small labeled dataset (for initial training) and a larger unlabeled dataset.
2. Initial Training:
Trained a classification model (SVM and logistic regression) on the small labeled dataset and evaluated the performance of the initial model using accuracy.
3. Pseudo-labeling:
Used the trained model to predict labels for the larger unlabeled dataset. These predictions will serve as pseudo-labels. Which will be combined with the labeled data to create a larger training set.
4. Iterative Training:
Retrained the model on the combined dataset (labeled + pseudo-labeled) and evaluated the performance of the updated model.

TODO:
- Repeat the process of pseudo-labeling, model training, and evaluation iteratively until satisfactory performance is achieved or until the performance plateaus.
- Fine-tune
