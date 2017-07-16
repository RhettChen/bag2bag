import tensorflow as tf 
import numpy as np 
import json

class BAGNNBuilder(object):
	def __init__(self, batch_size, learning_rate, embeddings, eos, lstm_units, sentence_length, clip, train = True, train_embeddings=False):
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.vocab_size = embeddings.shape[0]
		self.em_dim = embeddings.shape[1]
		self.embeddings_array = embeddings
		self.eos = eos # index of eos in the embeddings_array
		self.lstm_units = lstm_units
		self.sentence_length = sentence_length
		self.clip_value = clip
		self.train = train
		self.train_embeddings = train_embeddings

	def build_training_graph(self):
		#self.broken = tf.placeholder(dtype=tf.float32, shape=[None, self.sentence_length], name="broken_source")
		self.target = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.sentence_length], name="target_source")
		self.dropout_keep = tf.placeholder(dtype= tf.float32,name='dropout_keep')

		self.global_step = tf.Variable(0, name="global_step",trainable=False)
		broken = tf.nn.dropout(self.target,self.dropout_keep)*self.dropout_keep
		broken = tf.cast(self.target, tf.int32)
		self.embeddings = tf.Variable(self.embeddings_array, dtype=tf.float32,name="embeddings", trainable=self.train_embeddings)

		self.broken_embedded = tf.nn.embedding_lookup(self.embeddings, broken)
		#self.broken_embedded = tf.nn.dropout(self.broken_embedded,tf.cast(self.dropout_keep, tf.float64))
		######## rnn for encode #########
		initializer = tf.contrib.layers.xavier_initializer()
		self.lstm_fw = tf.contrib.rnn.LSTMCell(self.lstm_units, initializer=initializer)
		self.encoded_outputs_fw, self.encoded_state = tf.nn.dynamic_rnn(self.lstm_fw, self.broken_embedded, dtype = tf.float32)
		encoded_state_fw = self.encoded_state	
		encoded_list = encoded_state_fw

		if self.train:
			#seq2seq function  comparation
			list_input = self._tensor_to_list(self.broken_embedded)

			eos_batch = self._generate_batch_eos(self.sentence_length )
			embedded_eos = tf.nn.embedding_lookup(self.embeddings, eos_batch)

			#decoder_input = [embedded_eos] + list_input
			decoder_input = list_input
			self.lstm_decode = tf.contrib.rnn.LSTMCell(self.lstm_units, initializer=initializer)
			decoder_outputs_fw, _ = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_input, encoded_list, self.lstm_decode)
			# raw_output with shape (batch_size, length, lstm_units)
			raw_output = tf.reshape(decoder_outputs_fw, [self.batch_size,-1,self.lstm_units])
			dropout = tf.nn.dropout(raw_output, self.dropout_keep)
			self.decoder_output = dropout
		######### set projection from decoder_output to sentence#########
		self.projection_w = tf.Variable(tf.random_uniform([self.lstm_units, self.vocab_size],minval=-1,maxval=1), name="projection_weight")
		self.projection_b = tf.Variable(tf.zeros((self.vocab_size,)), name="projection_bias")
		if self.train:
			self._create_training_tensors()

	def train_BAGNN(self,sess, target, dropout_keep):
		feed_dict = {self.target:target, self.dropout_keep:dropout_keep}
		_, loss, w,b, out= sess.run([self.train_op,self.loss,self.projection_w,self.projection_b,self.decoder_output],feed_dict=feed_dict)
		return loss, w ,b, out


	def _create_training_tensors(self):
		sentence_as_list = self._tensor_to_list(tf.cast(self.target,tf.float32))
		eos_batch = self._generate_batch_eos(sentence_as_list[0])
		decoder_labels = sentence_as_list #+ [eos_batch]
		decoder_labels = [tf.cast(step, tf.int32) for step in decoder_labels]
		decoder_labels = tf.reshape(decoder_labels,[self.batch_size,self.sentence_length])

		# set the importance of each time step
		# 1 if before sentence end or EOS itself; 0 otherwise
		label_weights = [tf.cast(tf.less(i + 1, self.sentence_length), tf.float32) for i in range(self.sentence_length )] 
		lt = []
		for i in range(self.batch_size):
			lt.append(label_weights)
		label_weights = tf.reshape(lt,[self.batch_size,-1])
		projection_w_t = tf.transpose(self.projection_w)
		def loss_function(labels,inputs):
			labels = tf.reshape(labels, (-1,1))
			return tf.nn.sampled_softmax_loss(projection_w_t, self.projection_b,labels,inputs, 100, self.vocab_size)

		labeled_loss = tf.contrib.seq2seq.sequence_loss(self.decoder_output, decoder_labels,label_weights, softmax_loss_function= loss_function)
		self.loss = labeled_loss
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		gradients, v = zip(*optimizer.compute_gradients(self.loss))
		gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)


		self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                  global_step=self.global_step)

	def _tensor_to_list(self, tensor):
		return [tf.squeeze(step, [1]) for step in tf.split(tensor, self.sentence_length, 1)]

	def _generate_batch_eos(self, like):
		ones = tf.ones_like(like)
		return ones * self.eos

def Debug():
	with open("../Data/subset/idx.txt", "r") as data:
		data = json.loads(data.read())
	with open("../Data/subset/word2vec.txt","r") as vocab:
		vocab = json.loads(vocab.read())
	data = np.array(data)
	data = np.cast[np.float32](data)
	vocab = np.array(vocab)
	batch_size, sentence_length = data.shape[0],data.shape[1]
	learning_rate = 0.01
	eos = 1
	clip = 0.2
	iteration = 10000
	dropout_keep = 0.6
	model = BAGNNBuilder(batch_size,learning_rate,vocab,eos,100,sentence_length,clip)
	model.build_training_graph()
	total_loss = 0.0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(iteration):
			loss,w,b, out = model.train_BAGNN(sess,data,dropout_keep)
			total_loss += loss
			if(epoch%100==0):
				if(epoch!=0):
					print("epoch is {} \t loss is {}".format(epoch,total_loss/100))
				else:
					print("epoch is {} \t loss is {}".format(epoch,total_loss))
				total_loss = 0.0



Debug()


