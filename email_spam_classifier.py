'''
Email Spam Classifier (TF-IDF + MultinomialNB)

This script:
	•	Loads data/spam_ham_dataset.csv with columns: text, label_num.
	•	Cleans email text (remove HTML, normalize spaces) inside a Pipeline.
	•	Builds scikit-learn Pipeline: cleaner → TfidfVectorizer (1–2 n-grams, 20k feats, sublinear TF) → MultinomialNB.
	•	Runs 3-fold Stratified cross-validation with metrics: accuracy, precision, recall, F1.
	•	Saves metrics to reports/email_spam_multinomial.txt.
	•	Optionally prints a small random sample with TRUE vs PRED labels.
	•	Saves the trained pipeline to models/email_classifier_multinomial_model.joblib.

Notes:
	•	Label convention assumed: 1 = SPAM, 0 = HAM (update if different).

Usage:
python spam_classifier.py
'''

#Import libraries
#Import re
import re
#Import OS
import os
#Import joblib
from joblib import dump
#Import Pandas
import pandas as pd
#Import pipeline
from sklearn.pipeline import Pipeline
#Import function transformer
from sklearn.preprocessing import FunctionTransformer
#Import vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#Import model
from sklearn.naive_bayes import MultinomialNB
#Import cross-validate
from sklearn.model_selection import cross_validate, StratifiedKFold

#Import data
DATA_PATH = "data/spam_ham_dataset.csv"
email_df = pd.read_csv(DATA_PATH, encoding="latin-1")
email_df = email_df[["text", "label_num"]]
#Clean email text function
def clean_email_text(emails):
    '''Clean input email. Remove HTML tags, strip spaces'''
    clean_emails = []
    for email in emails:
        email = email.strip()
        email = re.sub(r"<.*?>", "", email)
        email = re.sub(r"[^A-Za-z0-9'\s]", "", email)
        clean_emails.append(email)
    return clean_emails
#Create pipeline
email_pipeline = Pipeline([
    ("cleaner", FunctionTransformer(
        clean_email_text,
        validate=False
    )),
    ("vectorizer", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        lowercase=True,
        stop_words="english",
        sublinear_tf=True
    )),
    ("multinomial_model", MultinomialNB())
])
#Cross-Validate
CV = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=1
)

SCORING = ["accuracy", "precision", "recall", "f1"]

email_cross_validate_results = cross_validate(
    email_pipeline,
    email_df["text"],
    email_df["label_num"],
    scoring=SCORING,
    cv=CV,
    return_train_score=False
)
#Create folder
os.makedirs("reports", exist_ok=True)
#Create report
report_lines = []
report_lines.append("Email Spam – MultinomialNB (Cross-Validation)\n")
report_lines.append(f"Folds: {CV.get_n_splits()}\n\n")

for metric in SCORING:
    scores = email_cross_validate_results[f"test_{metric}"]
    mean = scores.mean()
    std = scores.std()
    report_lines.append(
        f"{metric.capitalize():<10} mean±std: {mean:.4f} ± {std:.4f}  -> {scores}\n"
    )
# Write all metrics to a new report file
REPORT_PATH = "reports/email_spam_multinomial.txt"
with open(REPORT_PATH, mode="w", encoding="utf-8") as report:
    report.writelines(report_lines)

#Print categorized email
SAMPLE_EMAIL = True
N= 3
if SAMPLE_EMAIL:
    sample = email_df.sample(N, random_state=1)
    email_pipeline.fit(email_df["text"], email_df["label_num"])
    email_prediction = email_pipeline.predict(sample["text"])
    for email_text, true_label, predicted_label in zip(
        sample["text"],
        sample["label_num"],
        email_prediction
        ):
        TRUE = "SPAM" if true_label == 1 else "HAM"
        PRED = "SPAM" if predicted_label == 1 else "HAM"
        print(f"TRUE: {TRUE:<4} | PRED: {PRED:<4} -> {email_text[:80]}...")


os.makedirs("models", exist_ok=True)
PIPELINE_PATH = "models/email_classifer_multinomial_model.joblib"
dump(email_pipeline, PIPELINE_PATH)
