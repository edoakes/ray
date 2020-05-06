from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from joblib import dump
import pandas as pd

def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')
    dump(clf, 'classifiers/unigram_tf_idf.joblib')

imdb_train = pd.read_csv('csv/imdb_train.csv')
X_train = load_npz('vectorized_data/X_train_unigram_tf_idf.npz')
y_train = imdb_train['label'].values
train_and_show_scores(X_train, y_train, 'Unigram Tf-Idf')
