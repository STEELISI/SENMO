import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import read_df, label_cols
from metric import accuracy, true_positive, false_positive, \
					none_accuracy, pernote_accuracy

def build_documents(df_train):
	documents = []
	for col in label_cols:
		notes = df_train[df_train[col] == 1]["Note"]
		doc = ' '.join([x.strip() for x in notes])
		documents.append(doc)
	return documents

def train(df_train, X_train, vectorizer):
	classifiers = {}
	for col in label_cols:
		Y_train = df_train[col]
		binary_classifier = MultinomialNB()
		binary_classifier.fit(X_train, Y_train)
		classifiers[col] = binary_classifier

	return classifiers

def test(df_test, X_test, vectorizer, classifiers, pred_path):
	### PREDICTION ####
	Y_pred = {}
	for col in label_cols:
		pred = classifiers[col].predict(X_test)
		Y_pred[col] = pred

	df_preds = pd.DataFrame(data=Y_pred)
	df_preds.insert(0, "Note", df_test["Note"])

	### EVALUATION ###
	with open(pred_path + "/nb_score.txt", 'w') as score_file:
		print('### Accuracy ###')
		accuracy(df_preds, df_test, score_file)
		print('### True Positive ###')
		true_positive(df_preds, df_test, score_file)
		print('### False Positive ###')
		false_positive(df_preds, df_test, score_file)
		print('### None accuracy ###')
		none_accuracy(df_preds, df_test, score_file)
		print('### Per note accuracy ###')
		pernote_accuracy(df_preds, df_test, score_file)
	### SAVE PREDICTION ###
	df_preds.to_csv(pred_path + "/nb_pred.csv", index=False)

def nb(args):
	### READ INPUT ARGUMENTS
	train_path = args.i
	test_path = args.t
	pred_path = args.o
	feature_type = args.f
	### READ TRAINING DATA ###
	df_train = read_df(train_path)
	### READ TESTING DATA ###
	df_test = read_df(test_path)

	### FEATURIZATION ###
	if feature_type == "BOW":
		vectorizer = CountVectorizer()
		X_train = vectorizer.fit_transform(df_train["Note"])
		X_test = vectorizer.transform(df_test["Note"])
	elif feature_type == "TF-IDF":
		documents = build_documents(df_train)
		vectorizer = TfidfVectorizer()
		vectorizer.fit_transform(documents)
		X_train = vectorizer.transform(df_train["Note"])
		X_test = vectorizer.transform(df_test["Note"])

	### TRAINING ###
	classifiers = train(df_train, X_train, vectorizer)

	### TESTING ####
	test(df_test, X_test, vectorizer, classifiers, pred_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Training arguments",
		fromfile_prefix_chars='@')
	parser.add_argument("-i", type=str, help="path to input training file")
	parser.add_argument("-t", type=str, help="path to input testing file")
	parser.add_argument("-o", type=str, help="path to output prediction file")
	parser.add_argument("-f", type=str, help="feature types: BOW or TF-IDF")
	args = parser.parse_args()
	nb(args)
