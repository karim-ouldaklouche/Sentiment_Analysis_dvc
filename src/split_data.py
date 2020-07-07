import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split

import sys
import os

def split_data():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    dataset = os.path.join(input_dir, 'aclImdb_prepro.csv')

    trainset_out = os.path.join(output_dir, 'trainset.csv')
    testset_out = os.path.join(output_dir, 'testset.csv')

    dataset_df = pd.read_csv(dataset)

    # Split the dataset 
    trainset_df, testset_df = train_test_split(dataset_df, test_size = 0.3, random_state=888)

    # Write the data
    os.mkdir(os.path.join(output_dir))

    trainset_df.to_csv(trainset_out, index=False)
    testset_df.to_csv(testset_out, index=False)

if __name__ == "__main__":
    split_data()
    
