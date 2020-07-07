# The sentiment analysis Imdb movies with DVC

This repository has almost the same content  as the reprository Sentiment_Analysis. It uses DVC tool build on git (https://dvc.org/doc/install).

Also, the version 3.7 of python has been used.

A pipeline has been created with differents stages in order to create baseline models (Logistic regression nd SVM) with am embedding for the text from spacy : 

- prepare
- preprocess
- split_data
- embedding
- train_logreg
- train_svm 
- evaluate_logreg
- evaluate_svm

Two types of stage have not been included : analyse stage (this stage is not linked with the others) and prediction (predict_logreg and predict_svm)

## INSTALLATION

A virtual environement can be used for the case. 

The differents libraries are listed in the requirements.txt file.

```console
python -m venv sent_analysis_env
.\sent_analysis_env\Scripts\activate
pip install -r src/requirements.txt
```

```console
conda create -n sent_analysis_env python=3.7. anaconda
conda activate sent_analysis_env
pip install -r src/requirements.txt
```

## INITIALIZATION

Initialization 

```console
git init 
dvc init 
git commit -m "Initialize sentiment analysis Imdb movies"
```

## CONFIGURE

```console
dvc remote add -d sent_analysis_imdb_remote ...\dvc-storage (you can include your path)
git commit .dvc/config -m "Configure local remote"

dvc push
```	

## DATA

The differents dataset have been placed in the directory data. The scripts below, enables to create the folder and download the dataset.

```console
mkdir data
cd data

wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz --no-check-certificate

tar -xf aclImdb_v1.tar.gz

rm -f aclImdb_v1.tar.gz
```

## PREPARATION STAGE 

The stage merge the dataset from the opinion form Imdb movies.

```console
dvc run -f dvc/prepare.dvc -d src/prepare.py -d data/aclImdb -o data/prepared python src/prepare.py data/aclImdb data/prepared

git add dvc\prepare.dvc data\.gitignore
git commit -m "Create data preparation stage"
dvc push
```		

## PREPROCESSING TEXT

Stage of preprocessing the text
	
```console
dvc run -f dvc/preprocess.dvc -d src/preprocess.py -d data/prepared -o data/preprocess python src/preprocess.py data/prepared data/preprocess
	  
git add dvc/preprocess.dvc data\.gitignore
git commit -m "Preprocess stage"
dvc push
```

## SPLIT DATASET STAGE

Stage of split of the data in order to have a train and test set
	
```console
dvc run -f dvc/split_data.dvc -d src/split_data.py -d data/preprocess -o data/splited python src/split_data.py data/preprocess data/splited
	  
git add dvc\split_data.dvc data\.gitignore
git commit -m "Split dataset stage"
dvc push
```

## EMBEDDING STAGE

Stage of create the embedding for the train and test set
	
```console
dvc run -f dvc/embedding.dvc -d src/embedding.py -d data/preprocess -o data/embedding python src/embedding.py data/splited data/embedding
	  
git add dvc\embedding.dvc data\.gitignore
git commit -m "Embedding stage"
dvc push
```

## TRAIN MODELS STAGES

Stages of training with Logistic regression and SVM

```console			
dvc run -f dvc/train_logreg.dvc -d src/train_logreg.py -d data/splited -d data/embedding -o model/model_logreg.pkl -M metric/train.acc.metric.logreg python src/train_logreg.py data/splited data/embedding model/model_logreg.pkl metric/train.acc.metric.logreg

git add model\.gitignore dvc\train_logreg.dvc metric\train.acc.metric.logreg
git commit -m "Create training stages for logistic regression"
dvc push
```

```console			
dvc run -f dvc/train_svm.dvc -d src/train_svm.py -d data/splited -d data/embedding -o model/model_svm.pkl -M metric/train.acc.metric.svm python src/train_svm.py data/splited data/embedding model/model_svm.pkl metric/train.acc.metric.svm

git add dvc\train_svm.dvc metric\train.acc.metric.svm
git commit -m "Create training stages for SVM"
dvc push
```
					
## EVALUATE MODELS  STAGES

Evluation of the differents models 

```console		  
dvc run -f dvc/evaluate_logreg.dvc -d src/evaluate_logreg.py -d model/model_logreg.pkl -d data/splited -d data/embedding -M metric/test.acc.metric.logreg python src/evaluate_logreg.py model/model_logreg.pkl data/splited data/embedding metric/test.acc.metric.logreg
	
git add dvc\evaluate_logreg.dvc metric\test.acc.metric.logreg
git commit -m "Evaluation logistic regression model"
dvc push
```

```console		  
dvc run -f dvc/evaluate_svm.dvc -d src/evaluate_svm.py -d model/model_svm.pkl -d data/splited -d data/embedding -M metric/test.acc.metric.svm python src/evaluate_svm.py model/model_svm.pkl data/splited data/embedding metric/test.acc.metric.svm
	
git add dvc\evaluate_svm.dvc metric\test.acc.metric.svm
git commit -m "Evaluation SVM model"
dvc push
```

## THE METRICS
  
```console	
dvc metrics show
```

## RE EXECUTE THE STAGE 

```console
dvc repro dvc/evaluate_logreg.dvc

dvc repro dvc/evaluate_svm.dvc
```
