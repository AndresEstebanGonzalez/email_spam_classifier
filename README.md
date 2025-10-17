Email Spam Classifier (TF-IDF + MultinomialNB)

A simple, reproducible baseline to classify emails as SPAM or HAM using scikit-learn’s Pipeline with TF-IDF features and Multinomial Naive Bayes. Cross-validation metrics are written to a report file, and the trained pipeline is saved for reuse.

Features
	•	Clean text (remove HTML, normalize spaces) inside the Pipeline
	•	TF-IDF with 1–2 n-grams, English stopwords, sublinear TF, 20k features
	•	MultinomialNB baseline model
	•	3-fold Stratified cross-validation (accuracy, precision, recall, F1)
	•	Metrics saved to reports/email_spam_multinomial.txt
	•	Optional preview of TRUE vs PRED on a small random sample
	•	Trained pipeline saved to models/email_classifier_multinomial_model.joblib

Data

Place a CSV at data/spam_ham_dataset.csv with columns:
	•	text: raw email text
	•	label_num: numeric label (1 = SPAM, 0 = HAM)
(Update the label convention in code if your dataset differs.)

Quickstart
	1.	Create a virtual environment and install requirements:
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
	2.	Run the classifier:
python spam_classifier.py

Output
	•	Cross-validation metrics: reports/email_spam_multinomial.txt
Example:
Accuracy   mean±std: 0.97xx ± 0.00xx  -> [ … per-fold … ]
Precision  mean±std: …
Recall     mean±std: …
F1         mean±std: …
	•	Optional console sample:
TRUE: SPAM | PRED: SPAM -> Congratulations! You’ve won…
	•	Saved model:
models/email_classifier_multinomial_model.joblib

Use the Saved Model

Load and predict in another script or notebook:
from joblib import load
pipe = load(“models/email_classifier_multinomial_model.joblib”)
preds = pipe.predict([“Your free prize awaits!”, “See you at 3pm?”])

Repository Structure

data/
spam_ham_dataset.csv
models/
email_classifier_multinomial_model.joblib
reports/
email_spam_multinomial.txt
spam_classifier.py
requirements.txt
README.md

Requirements
	•	Python 3.10+
	•	scikit-learn
	•	pandas
	•	numpy
	•	joblib

Tip

If your dataset uses a different spam label, update the label mapping and the display logic accordingly.
