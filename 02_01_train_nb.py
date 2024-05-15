import pandas as pd
from datasets import load_dataset,load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
import joblib
from sklearn.feature_extraction import text

nltk.download('stopwords')
stops=list(set((stopwords.words('english'))))#+list(string.punctuation)+list(text.ENGLISH_STOP_WORDS))))
len(stops)

train_ds=load_from_disk('data/train_ds')
test_ds=load_from_disk('data/test_ds')

vectorizer = TfidfVectorizer(max_df=0.97, min_df=3, stop_words=stops)
nb_classifier = MultinomialNB()
pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('NB', nb_classifier)
])

y_train = pipeline.fit(train_ds['text'],np.array(train_ds['label']))
y_pred = pipeline.predict(test_ds['text'])

roc_auc = roc_auc_score(np.array(test_ds['label']), y_pred)
print(roc_auc)

joblib_file = "./models/nb_classifier_pipeline.joblib"
joblib.dump(pipeline, joblib_file)