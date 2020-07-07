import numpy as np
import pandas as pd

import unicodedata
import re
import os
import io
import sys

from pathlib import Path

import dask.dataframe as dd

import time

def unicode_to_ascii(s):
	"""
	Parameters : 
	text : String
	Returns
	text : String
	"""
	return ''.join(c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn')

def preprocess_text(text):
	"""
	Parameters : 
	text : String
	Returns
	text : String
	"""
	text = text.lower().strip()
	text = unicode_to_ascii(text)

	# creating a space between a word and the punctuation following it
	text = re.sub(r"([?.!,¿;])", r" \1 ", text)
	text = re.sub(r'[" "]+', " ", text)

	# replacing everything with space except (1-9, a-z, A-Z, ".", "?", "!", ",",";")
	text = re.sub(r"[^1-9a-zA-Z?.!,¿;]+", " ", text)

	return text

def preprocess():
	
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
	
	dataset = os.path.join(input_dir, 'aclImdb.csv')

	dataset_df = dd.read_csv(dataset)

	# """
	# Preprocessing of the data
	# """
	start = time.perf_counter()
	dataset_df['text_process'] = dataset_df['text'].apply(lambda x: preprocess_text(x))
	print(f'End process text in {time.perf_counter() - start} seconds')

	if not os.path.exists(output_dir):
		os.mkdir(os.path.join(output_dir))

	def name(i):
		return 'aclImdb_prepro'

	dataset_df.to_csv(output_dir + '/*.csv', 
						name_function=name, 
						index=False)

if __name__ == "__main__":
    preprocess()


