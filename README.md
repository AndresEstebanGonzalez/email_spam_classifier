⸻

Email Spam Classifier (TF-IDF + MultinomialNB)

A simple, reproducible baseline for classifying emails as SPAM or HAM using scikit-learn’s Pipeline with TF-IDF features and Multinomial Naive Bayes. Cross-validation metrics are written to a report file.

Features
	•	Clean text (remove HTML, normalize spaces) inside the Pipeline
	•	TF-IDF with 1–2 n-grams, English stopwords, sublinear TF
	•	MultinomialNB baseline
	•	3-fold Stratified cross-validation
	•	Metrics saved to reports/email_spam_multinomial.txt
	•	Optional preview of TRUE vs PRED on a small random sample

Data

Place a CSV at data/spam_ham_dataset.csv with columns:
	•	text: raw email text
	•	label_num: numeric label (1 = SPAM, 0 = HAM)
(Update the mapping in code if your dataset differs.)

Quickstart
	1.	Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Run the classifier:

python spam_classifier.py



Output
	•	Metrics report: reports/email_spam_multinomial.txt
Example content:

Accuracy   mean±std: 0.97xx ± 0.00xx  -> [ ... per-fold ... ]
Precision  mean±std: ...
Recall     mean±std: ...
F1         mean±std: ...


	•	Optional console sample:

TRUE: SPAM | PRED: SPAM -> Congratulations! You’ve won...



Repository Structure

data/
  spam_ham_dataset.csv
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

⸻
