import numpy as np
import pandas as pd 

import sys
import os

import pickle

import spacy

import time

def get_embedding(dataset, nlp, type_dataset, print_comment=False):
        """
        Parameters : 
        dataset : DataFrame
        type : String
        Returns
        embedding : List 
        """
        start = time.perf_counter()
        print(f'Start embedding for {type_dataset}set')
        embedding = []

        for index, row in dataset.iterrows():
            if print_comment:
                print('Index : ',index)
            embedding.append(nlp(str(row['text_process'])).vector)

        print(f'End embedding {type_dataset}set in {time.perf_counter()-start} seconds \n')
        return embedding

def create_embedding():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    trainset = os.path.join(input_dir, 'trainset.csv')
    testset = os.path.join(input_dir, 'testset.csv')

    trainset_out = os.path.join(output_dir, 'trainset_embedding.plk')
    testset_out = os.path.join(output_dir, 'testset_embedding.plk')

    trainset_df = pd.read_csv(trainset)
    testset_df = pd.read_csv(testset)

    os.mkdir(os.path.join(output_dir))

    """
    Embedding of the text with the small model with spacy
    """
    nlp=spacy.load("en_core_web_sm")

    trainset_embedding = get_embedding(trainset_df, nlp, type_dataset='train')

    with open(trainset_out, 'wb') as f:
            pickle.dump(trainset_embedding, f)

    testset_embedding = get_embedding(testset_df, nlp, type_dataset='test')

    with open(testset_out, 'wb') as f:
            pickle.dump(testset_embedding, f)

if __name__ == "__main__":
    create_embedding()  
    
