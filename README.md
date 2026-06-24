# 🚀 YouTube Spam Comment Detector using NLP & Machine Learning

An end-to-end Machine Learning and Natural Language Processing (NLP) project designed to automatically classify YouTube comments into **Ham** (legitimate comments) or **Spam** (promotional links, scams, or advertisements). 

This project demonstrates text preprocessing, vectorization, and the application of probabilistic classification models to solve real-world social media moderation challenges.

---

## 📊 Dataset Overview
The project utilizes a comprehensive dataset consisting of **5 distinct CSV files**, each containing comment logs from popular YouTube videos (such as Psy, Katy Perry, Eminem, Shakira, and LMFAO):
* `Youtube01.csv`
* `Youtube02.csv`
* `Youtube03.csv`
* `Youtube04.csv`
* `Youtube05.csv`

### Data Features:
* `COMMENT_ID`: Unique identifier for each comment.
* `AUTHOR`: The username of the commenter.
* `DATE`: Timestamp of the comment.
* `CONTENT`: The raw text content of the comment (The primary input feature).
* `CLASS`: The target label (`0` for Ham, `1` for Spam).

**Class Distribution:** Total of **1,955 comments** (1,004 Spam / 951 Ham), presenting a well-balanced dataset for robust training.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Manipulation:** `Pandas`, `NumPy`
* **Machine Learning & NLP:** `Scikit-Learn`
* **File Handling:** `Glob`, `Os`
* **Data Visualization:** `Matplotlib`, `Seaborn`

---

## 🔄 Detailed Project Workflow & Methodology

### 1. Data Integration & Preprocessing
* Used Python's `glob` library to dynamically read all 5 CSV files from the local directory and merged them into a single unified DataFrame using `pd.concat()`.
* Dropped metadata columns (`COMMENT_ID`, `AUTHOR`, `DATE`) that do not contribute to text semantics, isolating the `CONTENT` as the independent feature ($X$) and `CLASS` as the dependent target ($Y$).

### 2. Feature Extraction (Text Vectorization)
* Applied `CountVectorizer` from `sklearn` to convert the text comments into a matrix of token counts (**Bag-of-Words Model**). This extracts individual words, lowercases them, removes punctuation, and counts their frequencies across the corpus.

### 3. Dataset Splitting
* Split the processed dataset into **80% Training set** (to train the model) and **20% Testing set** (to evaluate unseen data).
* Utilized stratified splitting to ensure the original 51% Spam to 49% Ham ratio is perfectly preserved in both subsets.

### 4. Model Training & Classification
* Selected the **Multinomial Naive Bayes (`MultinomialNB`)** classifier. This algorithm is highly efficient and mathematically optimized for discrete features like word frequencies in text classification tasks based on Bayes' Theorem.

### 5. Model Evaluation
* Evaluated performance using a **Confusion Matrix** and a **Classification Report** to assess Precision, Recall, and F1-Score.

---

## 📈 Performance & Experimental Results

The trained Naive Bayes model achieved an outstanding **93% Overall Accuracy** on the testing dataset.

### Classification Report:
```text
              precision    recall  f1-score   support

         Ham       0.97      0.88      0.92       190
        Spam       0.89      0.98      0.93       201

    accuracy                           0.93       391
   macro avg       0.93      0.93      0.93       391
weighted avg       0.93      0.93      0.93       391
