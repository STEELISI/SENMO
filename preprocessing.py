import argparse
import os
import time
import re
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
# spell corrector libs
from textblob import TextBlob
from spellchecker import SpellChecker
from autocorrect import Speller
spellchecker = SpellChecker()
autocorrect = Speller(lang='en')

#stop words
stopwords = set()
PATH_TO_STOPWORDS_LIST = './STOPWORDS.txt'
with open(PATH_TO_STOPWORDS_LIST,'r') as fp:
	for l in fp:
		stopwords.add(l.strip())

"""
Convert all letters to lower or upper case (common : lower case)
"""
def convert_letters(tokens, style = "lower"):
	if (style == "lower"):
		return [token.lower() for token in tokens]
	else:
		return [token.upper() for token in tokens]

"""
Eliminate all continuous duplicate characters more than twice
"""
def reduce_lengthening(tokens):
	pattern = re.compile(r"(.)\1{2,}")
	return [pattern.sub(r"\1\1", token) for token in tokens]

"""
Stopwords Removal
"""
def remove_stopwords(tokens):
	return [token for token in tokens if token not in stopwords]

"""
Remove all digits and special characters
"""
def remove_special(tokens):
	return [re.sub("(\\d|\\W)+", " ", token) for token in tokens]

"""
Remove blancs on words
"""
def remove_blanc(tokens):
	return [token.strip() for token in tokens]

"""
Spell correction
"""
def spell_corrector(tokens, libr):
	if libr == 'regex':
		return reduce_lengthening(tokens)
	if libr == 'blob':
		return [str(TextBlob(token).correct()) for token in tokens]
	if libr == 'spellchecker':
		return [spellchecker.correction(token) for token in tokens]
	if libr == 'autocorrect':
		return [autocorrect(token) for token in tokens]

"""
Clean a note
"""
def clean(note, libr):
	tokens = nltk.word_tokenize(note)
	tokens = convert_letters(tokens)

	t1 = time.time()
	tokens = spell_corrector(tokens, libr)
	t2 = time.time()

	tokens = remove_stopwords(tokens)
	tokens = remove_special(tokens)
	tokens = remove_blanc(tokens)
	tokens = [t for t in tokens if len(t) != 0]
	note = ' '.join(tokens).strip()
	return note, float(t2 - t1)

"""
Main: PREPROCESSING
"""
def preprocessing(args):
	input_path = args.i
	output_path = args.o
	df = pd.read_csv(args.i)
	clean_notes = []
	dur = 0
	for i in range(len(df.index)):
		note, d = clean(df['Note'][i], args.c)
		clean_notes.append(note)
		dur += d
	print("lib: {}, time: {:3f}s".format(args.c, dur))
	df['Note'] = clean_notes
	df = df.sample(frac=1, random_state=1).reset_index(drop=True)
	directory = os.path.dirname(output_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	df.to_csv(output_path, index=False)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Preprocessing arguments",
		fromfile_prefix_chars='@')
	parser.add_argument("-i", type=str, help="path to input file")
	parser.add_argument("-o", type=str, help="path to output file")
	parser.add_argument("-c", type=str, 
		help="choice of spell corrector: regex, blob, spellchecker, autocorrect")
	args = parser.parse_args()
	preprocessing(args)
