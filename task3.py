"""
3. LSTM Model for Sentiment Classification

In this notebook, an RNN using an LSTM layer will be trained to classify movie reviews as positive or negative.

Notebook structure contains two main sections based on the version of the data. Each section has these subsections:

Part One: LSTM on Raw Text (unprocessed data directly from the original IMDB reviews)
  - Reading the Raw Data
  - Split the raw data
  - LSTM training on the raw data
  - Evaluations on the raw data

Part Two: LSTM on the Processed Set (the cleaned and lemmatized reviews from Task 1)
  - Reading the Processed Data
  - Split the processed data
  - LSTM training on the processed data
  - Evaluations on the processed data

The goal is to compare performance when using unprocessed versus processed inputs.

What “variable-length inputs” means:
Most neural networks expect inputs of the same size, but sentences and reviews naturally come in all lengths—from a few words to several paragraphs. “Variable-length input” simply means feeding each review at its own length (after tokenizing), rather than truncating or padding everything to a fixed size up front.

Final comparison of results shows trade-offs between overall accuracy and sensitivity to positive reviews.
"""

# Fix notebook seed to get same results whenever running the notebook again
import tensorflow as tf
SEED = 42
tf.random.set_seed(SEED)

# Common imports
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import Embedding, Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score
)
import seaborn as sns
from huggingface_hub import hf_hub_download

"""
Part One: LSTM on Raw Text
Reading the Raw Data from Hugging Face

Original raw data link:
https://drive.google.com/file/d/1vAcjI1BLEzjdqfSwcrwDWXIkw10DSgK6/view?usp=sharing
"""
raw_path = hf_hub_download(
    repo_id="LeenSMN/IMDB-50k-reviews",
    repo_type="dataset",
    filename="IMDB Dataset.csv"
)
df = pd.read_csv(raw_path)

# Split the raw data
X_raw = df['review']
y     = df['sentiment']

# 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# split 80% into 70% train, 10% val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125,
    random_state=SEED,
    stratify=y_temp
)

print(f"Train size: {len(X_train)} ({len(X_train)/len(df):.0%})")
print(f"Val   size: {len(X_val)}   ({len(X_val)/len(df):.0%})")
print(f"Test  size: {len(X_test)}  ({len(X_test)/len(df):.0%})")

# Plot distribution
fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), tight_layout=True)
for ax, (name, labels) in zip(axs, [("Train", y_train), ("Val", y_val), ("Test", y_test)]):
    counts = labels.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(name)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
plt.show()

"""
LSTM training on the Raw data

- Tokenizer: Keras built-in Tokenizer with num_words=5000, lower=True, oov_token='<OOV>'.
- Padding: pad to max_review_len with mask_zero=True in Embedding.
"""
top_words      = 5000
max_review_len = 500

tokenizer_drop = Tokenizer(num_words=top_words, lower=True, oov_token='<OOV>')
tokenizer_drop.fit_on_texts(X_train)

X_drop_train = pad_sequences(tokenizer_drop.texts_to_sequences(X_train), maxlen=max_review_len)
X_drop_val   = pad_sequences(tokenizer_drop.texts_to_sequences(X_val),   maxlen=max_review_len)
X_drop_test  = pad_sequences(tokenizer_drop.texts_to_sequences(X_test),  maxlen=max_review_len)

# Build dropout‐regularized LSTM
model_drop = Sequential([
    Embedding(top_words, 32, input_length=max_review_len),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model_drop.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Encode labels
label_map = {'negative':0, 'positive':1}
y_train_enc = y_train.map(label_map).values
y_val_enc   = y_val.map(label_map).values
y_test_enc  = y_test.map(label_map).values

# Train for 3 epochs to avoid overfitting
history_drop = model_drop.fit(
    X_drop_train, y_train_enc,
    validation_data=(X_drop_val, y_val_enc),
    epochs=3,
    batch_size=64,
    verbose=2
)

# Evaluations on the raw data
loss_drop, acc_drop = model_drop.evaluate(X_drop_test, y_test_enc, verbose=0)
print(f"Dropout Baseline Test Accuracy: {acc_drop*100:.2f}%")

# Detailed metrics
y_prob_drop = model_drop.predict(X_drop_test).ravel()
y_pred_drop = (y_prob_drop >= 0.5).astype(int)
cm_drop = confusion_matrix(y_test_enc, y_pred_drop)
tn, fp, fn, tp = cm_drop.ravel()

accuracy_drop = (tp + tn) / cm_drop.sum()
fpr_drop      = fp / (fp + tn)
fnr_drop      = fn / (fn + tp)
f1_drop       = f1_score(y_test_enc, y_pred_drop)
recall_drop   = recall_score(y_test_enc, y_pred_drop)

print(f"Accuracy:            {accuracy_drop:.3f}")
print(f"False Positive Rate: {fpr_drop:.3f}")
print(f"False Negative Rate: {fnr_drop:.3f}")
print(f"F1-score:            {f1_drop:.3f}")
print(f"Recall:              {recall_drop:.3f}")

sns.heatmap(cm_drop, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Pred_Neg','Pred_Pos'],
            yticklabels=['True_Neg','True_Pos'])
plt.title('Dropout Baseline Confusion Matrix (Raw Text)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""
Part Two: LSTM on the Processed Set

Reading the Processed Data from Hugging Face

Original processed data link:
https://drive.google.com/file/d/1PLoYC8owKyqwUpO9Td9mJC23JZ9zU1Cr/view?usp=sharing
"""
proc_path = hf_hub_download(
    repo_id="LeenSMN/IMDB-50k-reviews",
    repo_type="dataset",
    filename="preprocessed_reviews.csv"
)
df_processed = pd.read_csv(proc_path)

# Split processed data (same ratios & method as above)
X_raw = df_processed['review']
y     = df_processed['sentiment']

X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125,
    random_state=SEED,
    stratify=y_temp
)
print(f"Train size: {len(X_train)} ({len(X_train)/len(df_processed):.0%})")
print(f"Val   size: {len(X_val)}   ({len(X_val)/len(df_processed):.0%})")
print(f"Test  size: {len(X_test)}  ({len(X_test)/len(df_processed):.0%})")

fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), tight_layout=True)
for ax, (name, labels) in zip(axs, [("Train", y_train), ("Val", y_val), ("Test", y_test)]):
    counts = labels.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(name)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
plt.show()

"""
LSTM training on the processed data
(Architecture & training identical to raw-text section)
"""
tokenizer_drop.fit_on_texts(X_train)
X_drop_train = pad_sequences(tokenizer_drop.texts_to_sequences(X_train), maxlen=max_review_len)
X_drop_val   = pad_sequences(tokenizer_drop.texts_to_sequences(X_val),   maxlen=max_review_len)
X_drop_test  = pad_sequences(tokenizer_drop.texts_to_sequences(X_test),  maxlen=max_review_len)

history_drop = model_drop.fit(
    X_drop_train, y_train.map(label_map).values,
    validation_data=(X_drop_val, y_val.map(label_map).values),
    epochs=3,
    batch_size=64,
    verbose=2
)

loss_proc, acc_proc = model_drop.evaluate(X_drop_test, y_test.map(label_map).values, verbose=0)
print(f"Processed-text Dropout Baseline Test Accuracy: {acc_proc*100:.2f}%")

y_prob_proc = model_drop.predict(X_drop_test).ravel()
y_pred_proc = (y_prob_proc >= 0.5).astype(int)
cm_proc = confusion_matrix(y_test.map(label_map).values, y_pred_proc)
tn, fp, fn, tp = cm_proc.ravel()

accuracy_proc = (tp + tn) / cm_proc.sum()
fpr_proc      = fp / (fp + tn)
fnr_proc      = fn / (fn + tp)
f1_proc       = f1_score(y_test.map(label_map).values, y_pred_proc)
recall_proc   = recall_score(y_test.map(label_map).values, y_pred_proc)

print(f"Accuracy:            {accuracy_proc:.3f}")
print(f"False Positive Rate: {fpr_proc:.3f}")
print(f"False Negative Rate: {fnr_proc:.3f}")
print(f"F1-score:            {f1_proc:.3f}")
print(f"Recall:              {recall_proc:.3f}")

sns.heatmap(cm_proc, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Pred_Neg','Pred_Pos'],
            yticklabels=['True_Neg','True_Pos'])
plt.title('Processed-text Dropout Baseline Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Final comparison
import pandas as pd  # already imported above

metrics = {
    "Model": [
        "Raw-text Dropout Baseline",
        "Processed-text Dropout Baseline"
    ],
    "Accuracy": [accuracy_drop, accuracy_proc],
    "FPR":      [fpr_drop,      fpr_proc],
    "FNR":      [fnr_drop,      fnr_proc],
    "F1-score": [f1_drop,       f1_proc],
    "Recall":   [recall_drop,   recall_proc]
}

df_metrics = pd.DataFrame(metrics)
print(df_metrics)

"""
The comparison shows that the Raw-text Dropout Baseline achieves slightly higher accuracy and lower FPR but higher FNR and lower recall compared to the processed variant. The processed model trades a bit of accuracy for improved recall and a lower FNR, making it more sensitive to positive reviews. Choosing between them depends on whether minimizing overall errors or maximizing detection of positive cases is the priority.

After experimentation with deeper and bidirectional RNNs, the final dropout‐regularized LSTM at epoch 3 prevented overfitting best—further epochs increased validation loss and flattened accuracy, so training was stopped at epoch 3 for optimal generalization.
"""
