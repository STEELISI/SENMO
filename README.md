# SENMO
PETS 2022, Issue 3, **Paper #126: I know what you did on Venmo: Discovering privacy leaks in mobile social payments**.

This is the (partial) implementation of the paper.
SENMO is short for SENsitive content on venMO. 

## Overview
The SENMO pipeline consists of three runnning scripts.
1. [Preprocessing](#preprocessing): polish Venmo notes which are arbitrary to pure texts.
2. [Train](#train): use the cleaned text inputs with their labels, Trainset, to fine-tune BERT.
3. [Test](#test): evaluate the trained model on Testset and report scores.

## Requirements
To run our code, please install the dependency packages by using the following command: 
```
pip install -r requirements.txt
```
**NOTE**: We use a Conda environment with Python 3.9.10 to run the experiment. This code is tested and `requirements.txt` is generated for MAC M1 architecture. 
All packages can be installed by running `requirements.txt` except tensorflow (version 2.8.0). 
For MAC M1, please follow the Apple instructions [here](https://developer.apple.com/metal/tensorflow-plugin/) to install tensorflow.
For other platforms, tensorflow should be installed through the following Google instruction [here](https://www.tensorflow.org/install).

## Data
We store the dataset we use for training and testing the model in `./data/`. 
Specifically, inside `./data/`, we have `./data/train_orig.csv` or Trainset to fine-tune BERT and `./data/test_orig.csv` or Testset to evaluate the trained model.

## Preprocessing
Use the following command to preprocess the data.
```
python preprocessing.py @preprocessing.config
```
`preprocessing.py` contains several functions such as remove-stopword, remove-special-characters, remove-extra-whitespaces and spell-corrector. 
`preprocessing.config` define command line arguments that will be parsed to `preprocessing.py`. The format and arguments are illustrated below.
```
-i
./data/train_orig.csv
-o
./data/train_clean.csv
-c
regex
```
The format is simple. It contains a command line argument on the first line and its value on the next line.  
**Note: We use this format for all `.config` files to parse command line arguments and their values.**

We further explain the arguments specified above:
* -i: path to the labeled data in .csv format (please refer the format of labeled data on `./data/train_orig.csv` and `./data/test_orig.csv`).
* -o: path to save the preprocessed data.
* -c: choice of spell-corrector functions: 
"regex", 
"[blob](https://textblob.readthedocs.io/en/dev/)", 
"[spellchecker](https://pyspellchecker.readthedocs.io/en/latest/)", 
"[autocorrect](https://github.com/filyp/autocorrect)". 
We use "regex" in all the experiments presented on the paper. 
For more details about data preprocessing, please refer to Section 5.1 of the paper.

**Note**: we have to run `preprocessing.py` at least twice——one for `./data/train_orig.csv` and the other for `./data/test_orig.csv`.

## Train
We fine-tune the pre-trained language model [BERT](https://huggingface.co/docs/transformers/model_doc/bert) by the preprocessed Trainset from the previous step.  

To train the model, use the following command.
```
python train.py @train.config
```
We parse command line arguments specified in `train.config` (shown below) to `train.py`.
```
-i
./data/train_clean.csv
-o
./model/
-m
30
-b
32
-l
2e-5
-e
6
```
We further explain these arguments:
* -i: path to the preprocessed Trainset saved in the previous step.
* -o: path to a directory that the model weights will be saved.
* -m: max length or maximum number of tokens/words for each text input. In the paper, we set it to 30.
* -b: batch size. In the paper, we set it to 32.
* -l: learning rate. In the paper, we set it to 2e-5.
* -e: number of epochs. In the paper, we set it to 6.

For more details, please refer to Section 5.2 of the paper.

## Test
We evaluate the fine-tuned model from the previous step on the separate (preprocessed) Testset. 
To preprocess Testset, we run `preprocessing.py` by setting -i to `./data/test_orig.csv` and -o to `./data/test_clean.csv` (your choice).

Once the (preprocessed) Testset is ready, we run the following command to get the Testset predictions as well as evaluate the results.
```
python test.py @test.config
```
We parse command line arguments specified in `test.config` (shown below) to `test.py`.
```
-t
./data/test_clean.csv
-i
./model/
-o
./prediction/
-m
30
-b
32
```
We further explain these arguments:
* -t: path to the preprocessed Testset.
* -i: path to the directory that the model weights were saved in the previous step.
* -o: path to a directory that Testset predictions and evaluation results will be stored.
* -m: max length or maximum number of tokens/words for each text input (should be the same as specified in `train.config`).
* -b: number of epochs (should be the same as specified in `train.config`).

`test.py` will generated two output files: `pred.csv` and `score.txt` which will be saved in the directory specified by -o in `test.config`. 
`pred.csv` is the model predictions with the same format as Testset. `score.txt` contains several evaluation scores. 
Specifically, we report accuracy, true positive, false positive and per-note accuracy for every class. For more details, please refer to `metric.py`.

## Bugs or questions?
If you have any questions related to the code (i.e. run into problems while setting up dependencies or training/testing the model), 
feel free to email me, Pithayuth (charnset@usc.edu).
