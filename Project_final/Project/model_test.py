import tensorflow as tf
import os
import numpy as np
from scipy.spatial import distance
import csv

# load tensorflow model, this should be changed after train new model
LOG_DIR = os.path.abspath('..') + '\\log\\GloVec_1512190460\\'
META_FILE = 'my_GloVec.ckpt-16016.meta'

# restore our word vectors
def load_embeddings(meta_file=META_FILE,log_dir=LOG_DIR):
	print('Reading checkpoints...')
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(log_dir+meta_file)
		saver.restore(sess,tf.train.latest_checkpoint(log_dir))

		graph = tf.get_default_graph()
		embed_i = sess.run(graph.get_tensor_by_name('loss/embed_i:0'))
		embed_j = sess.run(graph.get_tensor_by_name('loss/embed_j:0'))

		final_embedding = embed_i 
		# print('word embedding:\n{}'.format(final_embedding))
		#print('embedding dimension:{}'.format(final_embedding.shape))

	return final_embedding

# test word-word similarity
def similarity_test(freq_dict,embedding,validation_pair=['china','japanese'],test_word='japanese'):
	for word in validation_pair:
		if word not in freq_dict:
			print('{} is not frequent!'.format(word))
			return None

	if test_word not in freq_dict:
		print('{} is not frequent!'.format(test_word))
		return None

	test_idx = freq_dict.index(test_word)
	test_embd = embedding[test_idx]

	val_idx0 = freq_dict.index(validation_pair[0])
	val_idx1 = freq_dict.index(validation_pair[1])

	val_embd0 = embedding[val_idx0]
	val_embd1 = embedding[val_idx1]

	X = val_embd0 - (val_embd1-test_embd)

	dist = []
	for i in range(embedding.shape[0]):
		if i == test_idx or i == val_idx0 or i == val_idx1:
			temp = np.inf
		else:
			temp = distance.cosine(X,embedding[i])
		dist.append(temp)

	result_idx = dist.index(min(dist))
	return freq_dict[result_idx]
	

# read csv file and return a list with each row in the csv file as an element
def read_csv(csv_path,delimiter='\n'):
	with open(csv_path) as csvfile:
		reader = csv.reader(csvfile,delimiter=delimiter)
		data = list(reader)
	return data

if __name__ == '__main__':

	dict_source = os.path.abspath('.') +'\\freq_wordlist.csv'
	freq_dict = read_csv(dict_source)

	flatten = [val for sublist in freq_dict for val in sublist]

	embedding = load_embeddings()

	# change validation_pair and test_word to test the result
	validation_pair = ['man','woman']
	test_word = 'queen'
	print('\n\n')
	print('Similarity test:')
	test_result = similarity_test(flatten,embedding,validation_pair,test_word)
	print('The word that is similar to "{}" in the same sense as "{}" is similar to "{}" is: "{}"'.format(
									test_word,validation_pair[0],validation_pair[1],test_result))
	print('\n')