import tensorflow as tf
import time 
import os
import pandas as pd
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

# get weight variables with name and shape
def weights(name,shape,stddev=0.02,trainable=True):
	weights = tf.get_variable(name,shape,tf.float32,trainable=trainable,initializer=tf.random_normal_initializer(stddev=stddev,dtype=tf.float32))
	return weights

# get bias variables with name and shape
def biases(name,shape,bias_start=0.0,trainable=True):
	biases = tf.get_variable(name,shape,tf.float32,trainable=trainable,initializer=tf.constant_initializer(bias_start,dtype=tf.float32))
	return biases

# the loss function of GloVe model
def loss(random_rows,random_cols,batch_X,batch_f_X,length,word_vector_length=100):
	'''
	random_rows: the row index of randomly selected nonzero values
	random_cols: the col index of randomly selected nonzero values
	batch_X: the selected nonzero values
	batch_f_X: the weighted result corresponding to X
	length: the length of rows(or columns) of co-occurrence matrix
	word_vector_length: the dimension of the word vector we want to obtain
	'''
	weight_i = weights('embed_i',[length,word_vector_length])
	weight_j = weights('embed_j',[length,word_vector_length])

	bias_i = biases('bias_i',[length])
	bias_j = biases('bias_j',[length])


	embed_i = tf.nn.embedding_lookup(weight_i,random_rows)
	embed_j = tf.nn.embedding_lookup(weight_j,random_cols)

	bi = tf.nn.embedding_lookup(bias_i,random_rows)
	bj = tf.nn.embedding_lookup(bias_j,random_cols)


	log_X = tf.log(batch_X)
	linear = tf.reduce_sum(tf.multiply(embed_i,embed_j),axis=1) + bi + bj

	factor = tf.square(linear-log_X)

	loss = tf.reduce_mean(tf.multiply(batch_f_X,factor))

	return loss,weight_i,weight_j,bias_i,bias_j


def train(X,f_X,batch_size=10,learning_rate=0.001,epoch=100,word_vector_length=100):
	global_step = tf.Variable(0,name='global_step',trainable=False)

	log_dir = os.path.abspath('.')
	
	length = X.shape[0]
	print(length)
	nonzero_rows,nonzero_cols = np.where(X > 0)
	len_nonzero = len(nonzero_rows)
	print(len_nonzero)

	h_random_rows = tf.placeholder(shape=[None,],dtype=tf.int64)
	h_random_cols = tf.placeholder(shape=[None,],dtype=tf.int64)
	h_batch_X = tf.placeholder(shape=[None,],dtype=tf.float32)
	h_batch_f_X = tf.placeholder(shape=[None,],dtype=tf.float32)
	

	with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):
		ls,weight_i,weight_j,bias_i,bias_j = loss(h_random_rows,h_random_cols,h_batch_X,h_batch_f_X,length,word_vector_length)

	with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(ls,global_step=global_step)

	loss_sum = tf.summary.scalar('loss',ls)
	weight_i_sum = tf.summary.histogram('weight_i',weight_i)
	weight_j_sum = tf.summary.histogram('weight_j',weight_j)

	bias_i_sum = tf.summary.histogram('bias_i',bias_i)
	bias_j_sum = tf.summary.histogram('bias_j',bias_j)

	merge = tf.summary.merge_all()

	# save the trained model
	saver = tf.train.Saver()
	model_name = 'GloVec_{}'.format(int(time.time()))

	# configurations for visualization
	config = projector.ProjectorConfig()
	embedding = config.embeddings.add()
	embedding.tensor_name = weight_i.name
	embedding.metadata_path = os.path.join(log_dir,'metadata.tsv')


	with tf.Session() as sess:
		writer = tf.summary.FileWriter(os.path.join(log_dir,model_name),sess.graph)
		projector.visualize_embeddings(writer,config)

		init = tf.global_variables_initializer()

		sess.run(init)

		iteration = len_nonzero//batch_size
		for epo in range(epoch):	
			for idx in range(iteration):
				# select random nonzero values
				random_indices = np.random.randint(len_nonzero,size=batch_size)
				random_rows = nonzero_rows[random_indices]
				random_cols = nonzero_cols[random_indices]

				batch_X = X[random_rows,random_cols]
				batch_f_X = f_X[random_rows,random_cols]

				_,summary_str,ls_value = sess.run([train_step,merge,ls],feed_dict={h_random_rows:random_rows,
																	   h_random_cols:random_cols,
																	   h_batch_X:batch_X,
																	   h_batch_f_X:batch_f_X})

				if idx % 20 == 0:
					print('epoch{}: [{}/{}] loss:{}'.format(epo,idx,iteration,ls_value))

			writer.add_summary(summary_str,epo+1)

			if epo % 10 == 0:
				checkpoint_path = os.path.join(os.path.join(log_dir,model_name),'my_GloVec.ckpt')
				saver.save(sess,checkpoint_path,global_step=global_step)

				print('===========    model saved    ===========' )

	with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):
		embedding_i = tf.get_variable('embed_i')
		embedding_j = tf.get_variable('embed_j')

	# return word vectors
	return embedding_i,embedding_j

# weight function f(X)
def weight_function(co_occurMat,x_max=100,a=np.float32(3/4)):
	shape = co_occurMat.shape
	result = np.ones(shape,dtype=np.float32)
	mask = co_occurMat < x_max
	result[mask] = np.power(co_occurMat[mask]/x_max,a)
	return result


if __name__ == '__main__':

	batch_size = 100000
	learning_rate = 0.001
	epoch = 100
	word_vector_length = 300

	co_occur_source = os.path.abspath('.') + '\\CPU_co_occurMat.csv'
	X = pd.read_csv(co_occur_source,sep=',',header=None).values
	print('co_occurMat load successfully!')
	f_X = weight_function(X)

	train(X,f_X,batch_size,learning_rate,epoch,word_vector_length)


