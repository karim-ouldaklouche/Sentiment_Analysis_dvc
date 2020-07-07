import pandas as pd

import sys
from sklearn.metrics import accuracy_score

import os
import pickle

def evaluate():

	model_file = sys.argv[1]
	splited_dir = sys.argv[2]
	embedding_dir = sys.argv[3]
	metrics_file = sys.argv[4]

	# load file
	testset = os.path.join(splited_dir, 'testset.csv')
	testset_df = pd.read_csv(testset)

	with open(os.path.join(embedding_dir, 'testset_embedding.plk'), 'rb') as fd:
		testset_embedding = pickle.load(fd)

	# Load the model
	with open(os.path.join(model_file), 'rb') as fd:
		model = pickle.load(fd)

    # Accuracy
	test_predict_lr = model.predict(testset_embedding)
	
	accuracy_score_test_lr = accuracy_score(test_predict_lr, testset_df['polarity'])
	print(accuracy_score_test_lr)

	# save the metric file
	with open(metrics_file, 'w') as fd:
	    fd.write('{:4f}\n'.format(accuracy_score_test_lr))

if __name__ == "__main__":
	evaluate()
