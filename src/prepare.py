import numpy as np 
import pandas as pd

import re
import sys
import os
import io
import random

import time

import glob

from pathlib import Path

from sklearn.utils import shuffle

import itertools

def parse_directory_for_files(path):
        """
        Parameters : 
        path : String
        Returns
        List 
        """
        print(f'Start parse directory for files : {path}')
        start = time.perf_counter()

        files = [f for f in Path(path).glob('**/*.txt')]
        print(f'End parse in {time.perf_counter() -start} seconds')

        return files

def read_files_to_dataframe(files, dataset_type, polarity):
        """
        Parameters : 
        files : List 
        polarity : String
        type : String
        Returns
        dataframe : DataFrame
        """
        start = time.perf_counter()
        print(f'start read {dataset_type} files with {polarity} polarity')
        rows = []
        for file in files:
            with open(file, mode='r', encoding="utf8") as f:
                rows.append({'text':f.read(), 'polarity':polarity, 'type':dataset_type})

        dataframe = pd.DataFrame(rows, columns=['text','polarity','type'])
        print(f'End read {dataset_type} files with {polarity} polarity in {time.perf_counter() -start} seconds')
        return dataframe

def prepare():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    output_file = os.path.join(output_dir, 'aclImdb.csv')

    dataset_types = ['train', 'test']
    polarities = ['pos','neg']

    dataset_types_polarities = [i for i in itertools.product(dataset_types, polarities)]

    files_types_polarities = []
    for types_polarity in dataset_types_polarities:
        files_type_polarity = parse_directory_for_files(os.path.join(input_dir,types_polarity[0],types_polarity[1]))
        files_types_polarities.append(files_type_polarity)

    files_to_dataframes = []
    for files,types_polarity in zip(files_types_polarities,dataset_types_polarities):
        file_to_dataframe = read_files_to_dataframe(files,types_polarity[0],types_polarity[1])
        files_to_dataframes.append(file_to_dataframe)

    dataset_all = pd.concat(files_to_dataframes)

    # Shuffle
    dataset_all = shuffle(dataset_all)

    # Write the new dataset in a file
    if not os.path.exists(output_dir):
        os.mkdir(os.path.join(output_dir))

    dataset_all.to_csv(output_file, index=False)

    print(dataset_all.shape)

if __name__ == "__main__":
    prepare()
