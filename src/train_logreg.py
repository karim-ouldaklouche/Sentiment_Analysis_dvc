import numpy as np
import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import sys

import pickle

import joblib

import os

import time

def train():	
	splited_dir = sys.argv[1]
	embedding_dir = sys.argv[2]
	model_output = sys.argv[3]
	metrics_file = sys.argv[4]

	if not os.path.exists('model'):
		os.mkdir(os.path.join('model'))

	if not os.path.exists('metric'):
		os.mkdir(os.path.join('metric'))

	# Load the files
	trainset = os.path.join(splited_dir, 'trainset.csv')
	trainset_df = pd.read_csv(trainset)

	with open(os.path.join(embedding_dir, 'trainset_embedding.plk'), 'rb') as fd:
		trainset_embedding = pickle.load(fd)

	# Train
	print('Start training')
  start = time.perf_counter()

	clf_lr = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')

	X = trainset_embedding
	y = trainset_df['polarity']
	clf_lr.fit(X, y)

	print(f'End training in {time.perf_counter()-start} seconds')

	with open(os.path.join(model_output), 'wb') as f:
            pickle.dump(clf_lr, f)

    # Accuracy
	train_predict_lr = clf_lr.predict(trainset_embedding)
	
	accuracy_score_train_lr = accuracy_score(train_predict_lr, trainset_df['polarity'])
	print(accuracy_score_train_lr)
	
	# save the metric file
	with open(metrics_file, 'w') as fd:
	    fd.write('{:4f}\n'.format(accuracy_score_train_lr))

if __name__ == "__main__":
	train()

