from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.sparse import save_npz, load_npz # used for saving and loading sparse matrices
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from os import system
import numpy as np
import pandas as pd
from joblib import dump, load

X_train = load_npz('vectorized_data/X_train_bigram_tf_idf.npz')
y_train = pd.read_csv('csv/imdb_train.csv')['label'].values

# Phase 2: penalty and alpha

clf = SGDClassifier()

distributions = dict(
        penalty=['l1', 'l2', 'elasticnet'],
        alpha=uniform(loc=1e-6, scale=1e-4)
)

random_search_cv = RandomizedSearchCV(
        estimator=clf,
        param_distributions=distributions,
        cv=5,
        n_iter=50
)
random_search_cv.fit(X_train, y_train)
print(f'Best params: {random_search_cv.best_params_}')
print(f'Best score: {random_search_cv.best_score_}')

# Save classifier.
system("mkdir 'classifiers'")

sgd_classifier = random_search_cv.best_estimator_

dump(random_search_cv.best_estimator_, 'classifiers/sgd_classifier.joblib')

# sgd_classifier = load('classifiers/sgd_classifier.joblib')
