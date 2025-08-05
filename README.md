# IMDB_Dataset_Sentiment_Classification_Task

# IMDB Dataset Sentiment Classification Task

This repository contains the task Python scripts, aligned with ProgressSoft Corporation’s assignment requirements and notebooks files that are the same as the scripts but are kept so they can be accessed from github for clarity.

# Repository Structure

- **notebooks/**
  - `Text_Processing.ipynb`
  - `ML_Model_for_Sentiment_Classification.ipynb`
  - `LSTM_for_Sentiment_Classification.ipynb`
  - `GPT2_Classification_using_Prompt_Engineering.ipynb`

- **scripts/**
  - `task1.py`  # Text processing  
  - `task2.py`  # ML model: TF–IDF + LinearSVC classification & evaluation  
  - `task3.py`  # LSTM classification on both raw & preprocessed data  
  - `task4.py`  # GPT-2 prompt-engineering classification (1-shot, 2-shot, 3-shot)

- **requirements.txt**  # All third-party dependencies

- **README.md**  # This file  


- **notebooks/**  
  Original `.ipynb` files demonstrating exploratory analysis, visualizations, and step-by-step development.

- **scripts/**  
  Clean, standalone `.py` scripts executable with a single Python interpreter, automatically downloading data from the Hugging Face Hub.

- **requirements.txt**  
  Lists all required packages; install via:
  ```bash
  pip install -r requirements.txt
````

## Assignment Overview

1. **Text Processing** (`task1.py`)

   * Load raw IMDB reviews
   * Lowercase, strip HTML tags, emails, URLs, punctuation
   * Remove stop words & lemmatize
   * Save processed reviews to a pickle file

2. **Machine Learning Classification** (`task2.py`)

   * Load preprocessed reviews
   * Train a TF–IDF + Linear SVC pipeline
   * Evaluate on hold-out test set (Accuracy, FPR, FNR, Precision/Recall/F1, ROC AUC, Balanced Accuracy)

3. **LSTM Classification** (`task3.py`)

   * Tokenize & pad sequences for both raw and preprocessed data
   * Build and train an LSTM network
   * Evaluate with the same set of metrics from task 2

4. **GPT-2 Prompt Engineering** (`task4.py`)

   * Use 1-shot, 2-shot, and 3-shot few-shot prompt approaches
   * Wrap the Hugging Face GPT-2 model in a simple CLI
   * Ensure outputs are strictly “Positive” or “Negative”

## Data

* **Source dataset:**
  [IMDB 50k Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Hosting:**
  Raw and preprocessed CSV files are hosted on the Hugging Face Hub at `LeenSMN/IMDB-50k-reviews` for automatic download.

## Notes

* The notebooks are included for transparency and educational clarity—they demonstrate exploratory steps, visualizations, and intermediate outputs.
* Only the `.py` scripts are required for assignment submission; they contain all runnable code in linear, stand-alone form.

## Acknowledgments

This work was completed as an assignment for ProgressSoft Corporation’s Apollo Team. All code was written manually without the assistance of LLM-based code generation tools.

```
::contentReference[oaicite:0]{index=0}
```
