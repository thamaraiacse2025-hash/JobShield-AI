import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# READ DATASET
# ==========================================

dataset = pd.read_csv("dataset/fake_job_postings.csv")

print("Dataset Shape:", dataset.shape)


# ==========================================
# FILL MISSING VALUES
# ==========================================

dataset = dataset.fillna("")


# ==========================================
# SELECT USEFUL COLUMNS
# ==========================================

dataset = dataset[
    [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "fraudulent",
    ]
]


# ==========================================
# COMBINE TEXT COLUMNS
# ==========================================

dataset["text"] = (
    dataset["title"] + " "
    + dataset["company_profile"] + " "
    + dataset["description"] + " "
    + dataset["requirements"] + " "
    + dataset["benefits"]
)


# ==========================================
# TF-IDF
# ==========================================

tfidf = TfidfVectorizer(stop_words="english")

X = tfidf.fit_transform(dataset["text"])


# ==========================================
# TARGET / OUTPUT
# ==========================================

Y = dataset["fraudulent"]


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# CREATE MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, Y_train)


# ==========================================
# PREDICT TEST DATA
# ==========================================

prediction = model.predict(X_test)


# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(Y_test, prediction)

precision = precision_score(
    Y_test,
    prediction,
    zero_division=0
)

recall = recall_score(
    Y_test,
    prediction,
    zero_division=0
)

f1 = f1_score(
    Y_test,
    prediction,
    zero_division=0
)

cm = confusion_matrix(
    Y_test,
    prediction
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("       MODEL EVALUATION")
print("===================================")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        Y_test,
        prediction,
        target_names=["Real Job", "Fake Job"],
        zero_division=0
    )
)

print("===================================")


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "model/fake_job_model.pkl"
)

joblib.dump(
    tfidf,
    "model/tfidf_vectorizer.pkl"
)

print("\nModel and TF-IDF Vectorizer Saved Successfully!")


# ==========================================
# TEST WITH NEW JOB POSTING
# ==========================================

new_job = """
We are hiring a software developer.
You will work with our development team to build web applications.
Candidates should have knowledge of Python and web development.
"""


# Convert new job using same TF-IDF
new_job_vector = tfidf.transform([new_job])


# Predict
result = model.predict(new_job_vector)


if result[0] == 1:
    print("\nNew Job Prediction: FAKE JOB")
else:
    print("\nNew Job Prediction: REAL JOB")

print("===================================")