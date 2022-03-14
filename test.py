import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFBertModel
from utils import read_df, get_ids_masks, create_dataset, BertClassifier, \
					label_cols, bert_model_name, get_keywords, not_in_keywords
from metric import accuracy, true_positive, false_positive, \
					none_accuracy, pernote_accuracy

def test(args):
	### READ INPUT ARGUMENTS
	test_path = args.t
	model_path = args.i
	pred_path = args.o
	if not os.path.exists(pred_path):
		os.makedirs(pred_path)
	MAX_LEN = int(args.m)
	BATCH_SIZE = int(args.b)
	### READ TRAINING DATA ###
	df = read_df(test_path)
	### EMBED TRAINING DATA ###
	input_ids, attention_masks = get_ids_masks(df["Note"], MAX_LEN)
	test_labels = df[label_cols].values
	### FINALIZE DATASET ###
	dataset = create_dataset((input_ids, attention_masks), 
		epochs=1, batch_size=BATCH_SIZE, train=False)
	### PREDICTION ###
	model = BertClassifier(TFBertModel.from_pretrained(bert_model_name), len(label_cols))
	model.load_weights(model_path)
	predictions = np.empty((0,len(label_cols)), int)
	for i, (token_ids, masks) in enumerate(dataset):
		start = i * BATCH_SIZE
		end = (i+1) * BATCH_SIZE - 1
		preds = model(token_ids, attention_mask=masks).numpy()
		binary_preds = np.where(preds > 0.5, 1, 0)
		predictions = np.append(predictions, binary_preds, axis=0)
	df_preds = pd.DataFrame(predictions, columns=label_cols)
	df_preds.insert(0, "Note", df["Note"])
	# ### KEYWORDS CHECKING ###
	if args.keywords:
		keywords = get_keywords()
		for index, row in df_preds.iterrows():
			if not_in_keywords(row['Note'], keywords):
				df_preds.loc[index, label_cols] = 0
	### EVALUATION ###
	with open(pred_path + "/score.txt", 'w') as score_file:
		print('### Accuracy ###')
		accuracy(df_preds, df, score_file)
		print('### True Positive ###')
		true_positive(df_preds, df, score_file)
		print('### False Positive ###')
		false_positive(df_preds, df, score_file)
		print('### None accuracy ###')
		none_accuracy(df_preds, df, score_file)
		print('### Per note accuracy ###')
		pernote_accuracy(df_preds, df, score_file)
	### SAVE PREDICTION ###
	df_preds.to_csv(pred_path + "/pred.csv", index=False)

if __name__ == "__main__":
	print("Num GPUs Available: ", 
		len(tf.config.experimental.list_physical_devices("GPU")))
	parser = argparse.ArgumentParser(description="Testing arguments",
		fromfile_prefix_chars='@')
	parser.add_argument("-t", type=str, help="path to input testing file")
	parser.add_argument("-i", type=str, help="path to input model file——dir of saved model's weights")
	parser.add_argument("-o", type=str, help="path to output prediction file")
	parser.add_argument("--keywords", action=argparse.BooleanOptionalAction, help="boolean: whether or not to use keyword filtering")
	parser.add_argument("-m", type=str, help="max length: should be the same as the training one")
	parser.add_argument("-b", type=str, help="batch size")
	args = parser.parse_args()
	test(args)
