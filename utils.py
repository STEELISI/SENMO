import time
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFBertModel
from transformers import BertTokenizer
from transformers import TFBertModel
from tensorflow.keras.layers import Dense, Flatten

bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
print('...BERT tokenizer loading complete')

cols_to_use = [
'Note',
'LGBTQ',
'ADULT_CONTENT',
'HEALTH',
'DRUGS_ALCOHOL_GAMBLING',
'RACE',
'VIOLENCE_CRIME',
'POLITICS',
'RELATION',
'LOCATION'
]
label_cols = cols_to_use[1:] #exclude note (input)

class BertClassifier(tf.keras.Model):    
	def __init__(self, bert: TFBertModel, num_classes: int):
		super().__init__()
		self.bert = bert
		self.classifier = Dense(num_classes, activation='sigmoid')

	@tf.function
	def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
		outputs = self.bert(input_ids,
							attention_mask=attention_mask,
							token_type_ids=token_type_ids,
							position_ids=position_ids,
							head_mask=head_mask)
		cls_output = outputs[1]
		cls_output = self.classifier(cls_output)

		return cls_output

def read_df(path):
	df = pd.read_csv(path)
	df = df[cols_to_use]
	print('Number of all sentences: {}'.format(len(df)))
	df['Note'] = df.Note.replace('NA',np.nan)
	df = df.dropna().sample(frac=1).reset_index(drop=True)
	print('Number of non-empty sentences: {}'.format(len(df)))
	return df

def get_ids_masks(sentences, MAX_LEN):
	ids = []
	masks = []
	for sent in sentences:
		encoded_dict = tokenizer.encode_plus(
			sent, # Sentence to encode.
			add_special_tokens = True, # Add '[CLS]' and '[SEP]'
			truncation = 'longest_first',
			max_length = MAX_LEN, # Pad & truncate all sentences.
			padding = 'max_length',
			return_attention_mask = True, # Construct attn. masks.
		)
		ids.append(encoded_dict['input_ids'])
		masks.append(encoded_dict['attention_mask'])
	return ids, masks

def create_dataset(data_tuple, epochs=1, batch_size=32, buffer_size=100, train=True):
	dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
	if train:
		dataset = dataset.shuffle(buffer_size=buffer_size)
	dataset = dataset.batch(batch_size)
	if train:
		dataset = dataset.prefetch(1)
	return dataset

