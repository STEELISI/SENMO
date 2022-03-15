import argparse
import os
import time
import re
import numpy as np
import pandas as pd
import nltk
import enchant
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

#===============================================================#
# spell corrector libs
from textblob import TextBlob
from spellchecker import SpellChecker
from autocorrect import Speller
spellchecker = SpellChecker()
autocorrect = Speller(lang='en')


#stop words
stopwords = set()
keywords = set()
d = enchant.Dict("en_US")
pattern = re.compile(r"(.)\1{2,}")
pattern_rm1 = re.compile(r"(.)\1{1,}")
email = re.compile("[^@]+@[^@]+\.[^@]+")
invc = re.compile("(((invoice|invc)(|s)|tracking)( \d|#|:| #| (\w)+\d))")
phno = re.compile("\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}")
acnt = re.compile("((password|passwd|paswd|pswd|pswrd|pwd|code|username|user name|userid|user id|: password )(:| is |: | : | to (my|his|her|their| the) |\? | \? |! |!! )([a-zA-Z0-9$#~!@%^&*]+))|((([a-zA-Z0-9$#~!@%^&*]+) is ((my|his|her|their| the) (password|passwd|paswd|pswd|pswrd|pwd|code|username|user name|userid|user id))))")
adr = re.compile("( (avenue|lane|road|boulevard|drive|street|ave|dr|rd|blvd|ln|st|way)(,|\.| ))|( (al|ak|as|az|ar|ca|co|ct|de|dc|fm|fl|ga|gu|hi|id|il|in|ia|ks|ky|la|me|mh|md|ma|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|mp|oh|ok|or|pw|pa|pr|ri|sc|sd|tn|tx|ut|vt|vi|va|wa|wv|wi|wy) \b\d{5}(?:-\d{4})?\b)")

#===============================================================#

PATH_TO_STOPWORDS_LIST = 'data/STOPWORDS.txt'
PATH_TO_KEYWORDS_LIST = "data/keyword_list.txt"
with open(PATH_TO_STOPWORDS_LIST,'r') as fp:
	for l in fp:
		stopwords.add(l.strip())
#===============================================================#
"""
Convert all letters to lower or upper case (common : lower case)
"""
def convert_letters(tokens, style = "lower"):
	if (style == "lower"):
		return [token.lower() for token in tokens]
	else:
		return [token.upper() for token in tokens]

#===============================================================#

with open(PATH_TO_KEYWORDS_LIST,'r') as fp:
        for l in fp:
                keywords.add(''.join(convert_letters(l.strip())))
#===============================================================#
"""
Eliminate all continuous duplicate characters more than twice
"""
def reduce_lengthening(tokens):
	t = []
	for token in tokens:
		if(d.check(token)):
			t.append(token)
		elif(d.check(pattern.sub(r"\1\1", token))):
			t.append(pattern.sub(r"\1\1", token))
		elif(d.check(pattern_rm1.sub(r"\1", token))):
			t.append(pattern_rm1.sub(r"\1", token))
		elif(token.lower() in keywords):
			t.append(token)
		elif(pattern.sub(r"\1\1",token.lower()) in keywords):
			t.append(pattern.sub(r"\1\1", token))
		elif(pattern_rm1.sub(r"\1", token.lower()) in keywords):  
			t.append(pattern_rm1.sub(r"\1", token))
		else:
			x = tree2conlltags(ne_chunk(pos_tag(word_tokenize(token))))
			x1 = tree2conlltags(ne_chunk(pos_tag(word_tokenize(pattern.sub(r"\1\1", token)))))
			x2 = tree2conlltags(ne_chunk(pos_tag(word_tokenize(pattern_rm1.sub(r"\1", token)))))
			if(len(x[0]) > 2 and ("B-" in x[0][2] or "I-" in x[0][2])):
				t.append(token)
			elif(len(x1[0]) > 2 and ("B-" in x1[0][2] or "I-" in x1[0][2])):
				t.append(pattern.sub(r"\1\1", token))
			elif(len(x2[0]) > 2 and ("B-" in x2[0][2] or "I-" in x2[0][2])):
				t.append(pattern_rm1.sub(r"\1", token))
			else:
				t.append(token)
				
	return t
#[pattern.sub(r"\1\1", token) for token in tokens]

#===============================================================#
"""
Eliminate all continuous duplicate characters more than once
"""
def reduce_lengthening_rm1(tokens):
    return [pattern_rm1.sub(r"\1", token) for token in tokens]
#===============================================================#
"""
Stopwords Removal
"""
def remove_stopwords(tokens):
	return [token for token in tokens if token not in stopwords]

#===============================================================#
"""
Remove all digits and special characters
"""
def remove_special(tokens):
	return [re.sub("(\\d|\\W)+", " ", token) for token in tokens]

#===============================================================#
"""
Remove blancs on words
"""
def remove_blanc(tokens):
	return [token.strip() for token in tokens]

#===============================================================#
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
#===============================================================#
"""
Clean a note
"""
def clean(note, libr):
	tokens = nltk.word_tokenize(note)
	#tokens = convert_letters(tokens)

	t1 = time.time()
	tokens = spell_corrector(tokens,libr)
	t2 = time.time()
	tokens = convert_letters(tokens)
	tokens = remove_stopwords(tokens)
	tokens = remove_special(tokens)
	tokens = remove_blanc(tokens)
	tokens = [t for t in tokens if len(t) != 0]
	note = ' '.join(tokens).strip()
	return note, float(t2 - t1)

#===============================================================#
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

#===============================================================#

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Preprocessing arguments",
		fromfile_prefix_chars='@')
	parser.add_argument("-i", type=str, help="path to input file")
	parser.add_argument("-o", type=str, help="path to output file")
	parser.add_argument("-c", type=str, 
		help="choice of spell corrector: regex, blob, spellchecker, autocorrect")
	args = parser.parse_args()
	preprocessing(args)
