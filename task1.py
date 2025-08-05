# Converted from Text_Processing.ipynb

# Converted from Text_Processing.ipynb

import pandas as pd  # To deal with DataFrames
import re, string
from huggingface_hub import hf_hub_download

# Download the raw dataset from Hugging Face
file_path = hf_hub_download(
    repo_id="LeenSMN/IMDB-50k-reviews",
    repo_type="dataset",
    filename="IMDB Dataset.csv"
)

#  EDA Section 
print("EDA Section ")
df = pd.read_csv(file_path)

print("First five rows:")
print(df.head(), "\n")

print("Data shape:", df.shape)
print("Data types:\n", df.dtypes, "\n")

print("Sentiment value counts:")
print(df['sentiment'].value_counts(), "\n")

print("Missing values per column:")
print(df[['review','sentiment']].isnull().sum(), "\n")

from collections import Counter

def top_n(texts, n=15):
    cnt = Counter()
    for t in texts:
        cnt.update(re.findall(r'\b\w+\b', t.lower()))
    return cnt.most_common(n)

for label in df['sentiment'].unique():
    print(f"Top words for {label}:")
    for word, freq in top_n(df[df['sentiment'] == label]['review']):
        print(f"  {word}: {freq}")
    print()

#  Text Processing Steps 
print("Text Processing Steps:")

# 1. Lowercase normalization
df['review'] = df['review'].str.lower()
print("Lowercase normalization done.")

# 2. Remove HTML tags
df['review'] = df['review'].str.replace(r'<[^>]+>', '', regex=True)
print("HTML tags removed.")

# 3. Remove email addresses
df['review'] = df['review'].replace(
    to_replace=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    value='',
    regex=True
)
print("Email addresses removed.")

# 4. Remove URLs
df['review'] = df['review'].str.replace(r'https?://\S+|www\.\S+', '', regex=True)
print("URLs removed.")

# 5. Remove punctuation
df['review'] = df['review'].str.translate(str.maketrans('', '', string.punctuation))
print("Punctuation removed.")

# 6. Remove stop words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop

df['review'] = df['review'].str.split().apply(
    lambda ws: ' '.join(w for w in ws if w.lower() not in stop)
)
print("Stop words removed.")

# 7. Lemmatization
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def spacy_lemma(text):
    return " ".join(tok.lemma_ for tok in nlp(text))

df['review'] = df['review'].apply(spacy_lemma)
print("Lemmatization done.")

# Save processed data to pickle
output_path = 'processed_reviews.pkl'
df.to_pickle(output_path)
print(f"Processed data saved to {output_path}")

# print that same review again to see the lemmatized version
print("Processing Text is Done and the processed data can be accessed from this link :https://drive.google.com/file/d/1vAcjI1BLEzjdqfSwcrwDWXIkw10DSgK6/view?usp=sharing")

#Display the first 5 rows
df.head()



