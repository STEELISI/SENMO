import argparse
import os
import tensorflow as tf
from transformers import TFBertModel
from utils import read_df, get_ids_masks, create_dataset, BertClassifier, \
					label_cols, bert_model_name

def train(args):
	### READ INPUT ARGUMENTS
	train_path = args.i
	model_path = args.o
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	MAX_LEN = int(args.m)
	BATCH_SIZE = int(args.b)
	LEARNING_RATE = float(args.l)
	NR_EPOCHS = int(args.e)
	### READ TRAINING DATA ###
	df = read_df(train_path)
	### EMBED TRAINING DATA ###
	input_ids, attention_masks = get_ids_masks(df["Note"], MAX_LEN)
	train_labels = df[label_cols].values
	### FINALIZE DATASET ###
	dataset = create_dataset((input_ids, attention_masks, train_labels), 
		epochs=1, batch_size=BATCH_SIZE, train=True)
	### TRAINING: INITIALIZATION ###
	model = BertClassifier(TFBertModel.from_pretrained(bert_model_name), len(label_cols))
	#-| Loss Function
	loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	#-| Optimizer (with 1-cycle-policy)
	optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-8)
	#-| Metrics
	train_auc_metrics = [tf.keras.metrics.AUC() for i in range(len(label_cols))]
	### TRAINING: LOOP ###
	print("BATCH: {}, EPOCHS: {}, LR: {}".format(BATCH_SIZE, NR_EPOCHS, LEARNING_RATE))
	for epoch in range(NR_EPOCHS):
		print('$$$$$$ EPOCH: {} $$$$$$'.format(str(epoch)))
		for i, (token_ids, masks, labels) in enumerate(dataset):
			labels = tf.dtypes.cast(labels, tf.float32)
			with tf.GradientTape() as tape:
				predictions = model(token_ids, attention_mask=masks)
				loss = loss_object(labels, predictions)
			#-| Backpropagation
			gradients = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(gradients, model.trainable_variables))
			train_loss(loss)
			for j, auc in enumerate(train_auc_metrics):
				auc.update_state(labels[:,j], predictions[:,j])
			#-| Progress report
			if i % 200 == 0:
				print(f'\nTrain Step: {i}, Loss: {train_loss.result()}')
				for i, label_name in enumerate(label_cols):
					print(f"{label_name} roc_auc {train_auc_metrics[i].result()}")
					train_auc_metrics[i].reset_states()
	### SAVE TRAINED MODEL ###
	model.save_weights(model_path)

if __name__ == "__main__":
	print("Num GPUs Available: ", 
		len(tf.config.experimental.list_physical_devices("GPU")))
	parser = argparse.ArgumentParser(description="Training arguments",
		fromfile_prefix_chars='@')
	parser.add_argument("-i", type=str, help="path to input training file")
	parser.add_argument("-o", type=str, help="path to output model file")
	parser.add_argument("-m", type=str, help="max length")
	parser.add_argument("-b", type=str, help="batch size")
	parser.add_argument("-l", type=str, help="learning rate")
	parser.add_argument("-e", type=str, help="number of epochs")
	args = parser.parse_args()
	train(args)
