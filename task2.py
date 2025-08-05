"""
2. Machine Learning Model for Sentiment Classification

In this notebook i:

- Load our preprocessed IMDB reviews
- Split the data for training
- Train on the training set using SVC
- Evaluation on the test set:
    • Accuracy
    • False Positive Rate (FPR)
    • False Negative Rate (FNR)
    • Precision, Recall, F1-score

Read the processed data from Hugging Face instead of Drive.
"""

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
)
import seaborn as sns
from huggingface_hub import hf_hub_download

# Download the preprocessed dataset from Hugging Face
file_path = hf_hub_download(
    repo_id="LeenSMN/IMDB-50k-reviews",
    repo_type="dataset",
    filename="preprocessed_reviews.csv"
)
print("Loaded preprocessed data from Hugging Face.")

# Read the data
df = pd.read_csv(file_path)

# Print the DataFrame shape and first few rows
print("Loaded DataFrame shape:", df.shape)
print(df.head(), "\n")

"""
Split the data using stratified split.
Stratified splitting ensures each subset (train/test)
maintains the same class proportions as the full dataset.
"""
X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print(f"Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

# Side-by-side bar charts for Train vs Test class distribution
fig, axs = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
for ax, (name, labels) in zip(axs, [("Train", y_train), ("Test", y_test)]):
    counts = labels.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"{name} Set")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
plt.show()

"""
Train SVC
To train SVC, TF–IDF was used because SVMs can only work with numbers, so
it converts each review into a fixed-length numeric vector that highlights
informative terms and down-weights overly common words.
"""
# Pipeline: TF–IDF → Linear SVC
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
    ("svc",   LinearSVC(C=1.0, max_iter=10000))
])

# Fit pipeline on training data
pipe.fit(X_train, y_train)
print("Model training complete.")

# Predict test labels on test set
y_pred = pipe.predict(X_test)

"""
Evaluations:
Required metrics:
  • Accuracy
  • False Positive Rate (FPR)
  • False Negative Rate (FNR)
  • Precision, Recall, F1-score
"""
# Compute confusion matrix entries and basic rates
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"Accuracy:            {accuracy:.3f}")
print(f"False Positive Rate: {fpr:.3f}")
print(f"False Negative Rate: {fnr:.3f}")

# Print full precision/recall/f1 report
print(classification_report(y_test, y_pred, digits=3))

# Plot confusion matrix as a heatmap
cm = [[tn, fp],
      [fn, tp]]
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Pred Neg','Pred Pos'],
    yticklabels=['True Neg','True Pos']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Other evaluations

# ROC AUC → turn the true labels into binary and feed in the SVM decision scores
y_binary = (y_test == "positive").astype(int)
y_scores = pipe.decision_function(X_test)
roc_auc = roc_auc_score(y_binary, y_scores)
print(f"ROC AUC: {roc_auc:.3f}")

# Balanced accuracy (average of recall for each class)
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {bal_acc:.3f}")
# Plot ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_binary, y_scores)
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.3f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()      

# Save the trained model to disk
import joblib
model_filename = "sentiment_model.joblib"
joblib.dump(pipe, model_filename)
print(f"Model saved to {model_filename}")   