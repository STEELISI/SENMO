# SENMO
PETS 2022, Issue 3, **Paper #126: I know what you did on Venmo: Discovering privacy leaks in mobile social payments**.

Authors: Rajat Tandon, Pithayuth Charnsethikul, Ishank Arora, Dhiraj Murthy, and Jelena Mirkovic

The classification framework SENMO (SENsitive content on venMO) classifies a Venmo transaction note as one or more of the sensitive categories from Table 1. A note could also be classified as NON (non-sensitive), if it does not contain any sensitive information.

This repository includes the basic version of the tools SENMO and SENMO-npre. The code here demonstrates the results presented in Table 13 of the paper for SENMO and SENMO-npre.

SENMO-npre: classifies Venmo notes as one or more of the sensitive categories from Table 1 using BERT without applying sensitive keywords pre-filters on the classification input (i.e, without using the list of known sensitive keywords).

SENMO: classifies Venmo notes using BERT as one or more of the sensitive categories from Table 1 after applying sensitive keywords pre-filters on the classification input (i.e, using the list of known sensitive keywords).


## Overview
The SENMO pipeline consists of running the scripts.
1. [Preprocessing](#preprocessing): cleans Venmo notes, deleting extra whitespaces, stopwords, emojis, etc.
2. [Train](#train): uses the cleaned text inputs with their labels, training set (fine-tuning set as per the paper), to fine-tune BERT.
3. [Test](#test): evaluates the trained model on the testing set (evaluation set as per the paper) and report the classification accuracy.

## Requirements
To run our code, please install the dependency packages by using the following command: 
```
pip install -r installation/requirements.txt
```
**Note**: We have tested our code on Python 3.7.7 and up. `requirements.txt` has been generated for macOS M1 architecture. The code should work or can be made to work on other platforms too. 
All packages can be installed by running `requirements.txt` except tensorflow (version 2.8.0). 
For MAC M1, please follow the Apple instructions [here](https://developer.apple.com/metal/tensorflow-plugin/) to install tensorflow.
For other platforms, tensorflow should be installed through the following Google instruction [here](https://www.tensorflow.org/install).

You may also run into some issues while installing the "pyenchant" package if enchant library is missing in the machine. Some useful links for fixing these issues are: [link-1](https://pyenchant.github.io/pyenchant/install.html),[link-2](https://github.com/pyenchant/pyenchant/issues/164) and [link-3](https://stackoverflow.com/questions/29381919/importerror-the-enchant-c-library-was-not-found-please-install-it-via-your-o).

## Data
We store the datasets that we use for training and testing the model in `./data/`. 
Specifically, inside `./data/`, we have `./data/train_clean.csv` or training set (fine-tuning set) to fine-tune BERT and `./data/test_clean.csv` or testing set (evaluation set) to evaluate the trained model.

Training set (fine-tuning set): About 1,000 notes per category LGB, ADU, HEA, DAG, POL, RET, VCR, REL, LOC, NON from the dataset D2, were manually labelled by us for fine-tuning BERT.

Testing set (evaluation set): About 100 notes per category LGB, ADU, HEA, DAG, POL, RET, VCR, REL, LOC, NON from the dataset D2, were manually labelled by us for creating this test set after removing the fine-tuning set entries from D2.

Additional details about these datasets are provided in the paper, Section 5.2.

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
We use "regex" in all the experiments presented in the paper. 
For more details about data preprocessing, please refer to Section 5.1 of the paper.

**Note**: we have to run `preprocessing.py` at least twice——one for `./data/train_orig.csv` and the other for `./data/test_orig.csv`.

## Train
We fine-tune the pre-trained language model [BERT](https://huggingface.co/docs/transformers/model_doc/bert) by the preprocessed training set from the previous step.  

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
* -i: path to the preprocessed training set saved in the previous step.
* -o: path to a directory that the model weights will be saved.
* -m: max length or maximum number of tokens/words for each text input. In the paper, we set it to 30.
* -b: batch size. In the paper, we set it to 32.
* -l: learning rate. In the paper, we set it to 2e-5.
* -e: number of epochs. In the paper, we set it to 6.

For more details, please refer to Section 5.2 of the paper.

## Test
We evaluate the fine-tuned model from the previous step on the separate (preprocessed) testing set. 
To preprocess testing set, we run `preprocessing.py` by setting -i to `./data/test_orig.csv` and -o to `./data/test_clean.csv` (your choice).

Once the (preprocessed) testing set is ready, we run the following command to get the testing set predictions as well as evaluate the results.
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
--keywords
```
We further explain these arguments:
* -t: path to the preprocessed testing set.
* -i: path to the directory that the model weights were saved in the previous step.
* -o: path to a directory that testing set predictions and evaluation results will be stored.
* -m: max length or maximum number of tokens/words for each text input (should be the same as specified in `train.config`).
* -b: number of epochs (should be the same as specified in `train.config`).
* --keywords: To run SENMO, i.e. to use the list of known sensitive keywords for pre-filtering

**Note**: To run SENMO-npre, delete the line "--keywords" from test.config file.

`test.py` will generate two output files: `pred.csv` and `score.txt` which will be saved in the directory specified by -o in `test.config`. 
`pred.csv` is the model predictions with the same format as testing set. `score.txt` contains several evaluation scores. 
Specifically, we report accuracy, true positive, false positive and per-note accuracy for every class. For more details, please refer to `metric.py`.

## Lexicon : List of sensitive keywords

We also release as open-source the list of sensitive keywords for the different sensitive categories present inside data/Lexicon folder. The details about the references that we use to prepare it is shown in Table 11 of the paper.  We build the list of keywords associated with sensitive content for different categories using various popular sources and prior published works.

## Questions?
If you have any questions related to the code (i.e. run into problems while setting up dependencies or training/testing the model), feel free to email us at: (rajattan@usc.edu) and (charnset@usc.edu).

**Note**: BERT's fine-tuned models are non-deterministic. Hence, we can get slightly different results every time we re-train or fine-tune BERT on the same data. We may not get the exact same results but approximately similar results. 

Also, the training set and testing set includes some duplicate notes too as different/same users post same notes too. For example, the note "marijuana" was posted by multiple users. Similarly, "For bailing me out of jail" is another such example.

The main reason behind the duplicate entries in fine-tuning and evaluation datasets is that Venmo has a very high presence of duplicates. For example, the publicly available Venmo dataset (D2 in the paper) which comprises ~7.1M notes, contains approximately ~62% duplicate entries, coming from different Venmo notes by different users.

Our classifier only applies to Venmo, since our fine-tuning/evaluation datasets reflect the composition and presence of duplicates in the original Venmo notes. We make no claims in the paper about the usefulness of this classifier to classify other sensitive content on other platforms. 
