# Converted from Text_Processing.ipynb

import pandas as pd #To deal with Dataframs
import re, string

# access the data from hugging face
from huggingface_hub import hf_hub_download
import pandas as pd

# file path on MyDrive
file_path =  hf_hub_download(
    repo_id="LeenSMN/IMDB-50k-reviews",
       repo_type="dataset", 
    filename="IMDB Dataset.csv"
)
print("EDA section is running...")
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

df.describe()

# Display the first five rows of the DataFrame
print(df.head())

# Displaying the data shape + type + count for each class
print("Shape:", df.shape)
print(df.dtypes)
print("\n")
print("Sentiment value counts:")
print(df['sentiment'].value_counts())


# Count missing values
print(df[['review','sentiment']].isnull().sum())

# Counting Top frequent words for each class
from collections import Counter
import re

def top_n(texts, n=15):
    cnt = Counter()
    for t in texts:
        cnt.update(re.findall(r'\b\w+\b', t.lower()))
    return cnt.most_common(n)

for label in df['sentiment'].unique():
    print(f"\nTop words for {label}:")
    for word, freq in top_n(df[df['sentiment']==label]['review']):
        print(f"  {word}: {freq}")

print("Text processing steps are running...")
#Normalize the text by making all letters lowercase.
df['review'] = df['review'].str.lower()
print("")

#Remove all HTML tags (e.g., <br/>).
#Using pandas Series.str.replace with a regex to strip (Delete) anything between <…>
df['review'] = df['review'].str.replace(r'<[^>]+>', '', regex=True)
print("")

# Print first 5 reviews containing email addresses
# Print full text
for idx, text in df.loc[df['review'].str.contains(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', na=False), 'review'].head(5).items():
    print("")

print("To lowercase normalization is done and HTML tags are removed. ")



#Remove all email addresses -> using regex
df['review'] = df['review'].replace(to_replace=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', value='', regex=True)



# Print first row that contains a URL
df.loc[df['review'].str.contains(r'https?://\S+|www\.\S+', na=False)].head(1)

# Remove all URLs from 'review'
df['review'] = df['review'].str.replace(r'https?://\S+|www\.\S+', '', regex=True)

print("URLs are removed from the reviews.")
# detect first row with punctuation and display it
idx = df.index[df['review'].str.contains(f"[{re.escape(string.punctuation)}]", na=False)][0]
print("")

#remove punctuation from 'review'
df['review'] = df['review'].str.translate(str.maketrans('', '', string.punctuation))

#show same row after punctuation removal
print("")
print("Punctuation is removed from the reviews.")
#detect first row containing stop words and display it -> Using scikit learn built-in English stop word list (ENGLISH_STOP_WORDS)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop
idx = df.index[df['review'].str.lower().str.split().apply(lambda ws: any(w in stop for w in ws))][0]
print("")

## remove stop words from 'review'
df['review'] = df['review'].str.split().apply(lambda ws: ' '.join(w for w in ws if w.lower() not in stop))

#show same row after stop words removal
print("")
print("Stop words are removed from the reviews.")
# get the first review that changes after spaCy lemmatization
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
def spacy_lemma(text):
    return " ".join(tok.lemma_ for tok in nlp(text))

#Locate the first review whose lemmatized version differs from the original
idx = next(i for i, t in df['review'].items() if spacy_lemma(t) != t)
#Show that index and its original review text
print("")


#  lemmatize every review using spaCy
df['review'] = df['review'].apply(spacy_lemma)
# print that same review again to see the lemmatized version
print("Processing Text is Done and the processed data can be accessed from this link :https://drive.google.com/file/d/1vAcjI1BLEzjdqfSwcrwDWXIkw10DSgK6/view?usp=sharing")

#Display the first 5 rows
df.head()



