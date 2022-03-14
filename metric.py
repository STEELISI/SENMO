from utils import label_cols

def accuracy(df_preds, df_test, output_file):
	for col in label_cols:
		c = 0
		n = len(df_test)
		for index, value in df_preds[col].items():
			if df_preds[col][index] == df_test[col][index]:
				c = c + 1
		# print
		print('{} accuracy : {}/{} = {}'.format(col, c, n, str(c/n)))
		# write
		output_file.write('{} accuracy : {}/{} = {}'.format(col, c, n, str(c/n)))
		output_file.write('\n')
	print('\n')
	output_file.write('\n')

def true_positive(df_preds, df_test, output_file):
	for col in label_cols:
		c = 0
		n = 0
		for index, value in df_preds[col].items():
			if df_test[col][index] == 1:
				n = n + 1
				if df_preds[col][index] == df_test[col][index]:
					c = c + 1
		# print
		print('=== {} ==='.format(col))
		print('true positive : {}/{} = {}'.format(c, n, str(c/n)))
		print('false negative : {}/{} = {}'.format(n - c, n, str(1 - c/n)))
		# write
		output_file.write('=== {} ==='.format(col))
		output_file.write('\n')
		output_file.write('true positive : {}/{} = {}'.format(c, n, str(c/n)))
		output_file.write('\n')
		output_file.write('false negative : {}/{} = {}'.format(n - c, n, str(1 - c/n)))
		output_file.write('\n')
	print('\n')
	output_file.write('\n')

def false_positive(df_preds, df_test, output_file):
	for col in label_cols:
		c = 0
		n = 0
		for index, value in df_preds[col].items():
			if df_test[col][index] == 0:
				n = n + 1
				if df_preds[col][index] == df_test[col][index]:
					c = c + 1
		# print
		print('=== {} ==='.format(col))
		print('true negative : {}/{} = {}'.format(c, n, str(c/n)))
		print('false positive : {}/{} = {}'.format(n - c, n, str(1 - c/n)))
		# write
		output_file.write('=== {} ==='.format(col))
		output_file.write('\n')
		output_file.write('true negative : {}/{} = {}'.format(c, n, str(c/n)))
		output_file.write('\n')
		output_file.write('false positive : {}/{} = {}'.format(n - c, n, str(1 - c/n)))
		output_file.write('\n')
	print('\n')
	output_file.write('\n')

### compute accuracy for NONE class"
def none_accuracy(df_preds, df_test, output_file):
	c1 = 0
	n1 = 0
	for index, row in df_test.iterrows():
		if row[label_cols].sum() == 0:
			n1 = n1 + 1
			if df_preds.iloc[index,1:].sum() == 0: # is 'NONE'
				c1 = c1 + 1
	# print
	print('=== NONE ===')
	print('true positive : {}/{} = {}'.format(c1, n1, str(c1/n1)))
	print('false negative : {}/{} = {}'.format(n1 - c1, n1, str(1 - c1/n1)))
	# write
	output_file.write('=== NONE ===')
	output_file.write('\n')
	output_file.write('true positive : {}/{} = {}'.format(c1, n1, str(c1/n1)))
	output_file.write('\n')
	output_file.write('false negative : {}/{} = {}'.format(n1 - c1, n1, str(1 - c1/n1)))
	output_file.write('\n')

	c2 = 0
	n2 = 0
	for index, row in df_test.iterrows():
		if row[label_cols].sum() != 0:
			n2 = n2 + 1
			if df_preds.iloc[index,1:].sum() != 0: # is not 'NONE'
				c2 = c2 + 1
	# print
	print('=== NONE ===')
	print('true negative : {}/{} = {}'.format(c2, n2, str(c2/n2)))
	print('false positive : {}/{} = {}'.format(n2 - c2, n2, str(1 - c2/n2)))
	print('\n')
	# write
	output_file.write('=== NONE ===')
	output_file.write('\n')
	output_file.write('true negative : {}/{} = {}'.format(c2, n2, str(c2/n2)))
	output_file.write('\n')
	output_file.write('false positive : {}/{} = {}'.format(n2 - c2, n2, str(1 - c2/n2)))
	output_file.write('\n')
	output_file.write('\n')

### per-note accuracy: accuracy that the model predicts every class correctly for each note ###
def pernote_accuracy(df_preds, df_test, output_file):
	c = 0
	n = len(df_test)
	for i in range(n):
		if df_preds.iloc[i,:].equals(df_test.iloc[i,:]):
			c = c + 1
	# print
	print('Per note accuracy : {}/{} = {}'.format(c, n, str(c/n)))
	print('\n')
	# write
	output_file.write('Per note accuracy : {}/{} = {}'.format(c, n, str(c/n)))
	output_file.write('\n')
	output_file.write('\n')