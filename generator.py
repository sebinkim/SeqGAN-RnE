import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq

class LSTMHelper(seq2seq.Helper) :
	def __init__(self, t_batch_size, emb_matrix, seq_len, given_len, given_seq, start_ids, W, b) :
		
		self.t_batch_size = t_batch_size
		self.emb_matrix = emb_matrix
		self.seq_len = seq_len
		self.given_len = given_len
		self.given_seq = given_seq
		self.W, self.b = W, b
		self.start_emb = tf.nn.embedding_lookup(self.emb_matrix, start_ids)
		
#		print(given_len, given_ids.shape)
		
	@property
	def batch_size(self):
		return self.t_batch_size
	
	@property
	def sample_ids_shape(self):
		return tf.TensorShape([])
	
	@property
	def sample_ids_dtype(self):
		return tf.int32
	
	def initialize(self) :
		finished = tf.tile([False], [self.t_batch_size])
		return (finished, self.start_emb)
	
	def sample(self, time, outputs, state) :
		prob = tf.log(tf.nn.softmax(tf.matmul(outputs, self.W) + self.b))
		sample_ids = tf.cast(tf.reshape(tf.random.categorical(prob, 1), [self.batch_size]), dtype = tf.int32)
		return sample_ids
		
	def next_inputs(self, time, outputs, state, sample_ids) :
		finished = tf.greater_equal(time + 1, self.seq_len)
		next_inputs = tf.cond(
			tf.greater_equal(time + 1, self.given_len),
			lambda: tf.nn.embedding_lookup(self.emb_matrix, sample_ids),
			lambda: tf.nn.embedding_lookup(self.emb_matrix, self.given_seq[:, time])
		)
		return (finished, next_inputs, state)
		
###################################################################################################

class Generator :
	def __init__(self, sess, batch_size, seq_len, vocab_size, emb_dim, hidden_layer, montecarlo_num, grad_clip, 
		pre_learning_rate, adv_learning_rate, montecarlo_k = None, montecarlo_p = None) :
		
		self.sess = sess
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.hidden_layer = hidden_layer
		self.montecarlo_num = montecarlo_num
		self.montecarlo_k = montecarlo_k
		self.montecarlo_p = montecarlo_p
		self.grad_clip = grad_clip
		self.pre_learning_rate = pre_learning_rate
		self.adv_learning_rate = adv_learning_rate
		self.train_vars = []
		
		self.given_seq = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
		self.rewards = tf.placeholder(tf.float32, [self.batch_size, self.seq_len])
		
		self.start_ids = tf.Variable(tf.tile([0], [self.batch_size]))
		
		self.emb_matrix = tf.Variable(tf.random_normal([self.vocab_size, self.emb_dim], stddev = 0.1))
		self.rnn_cell = rnn.LSTMCell(self.hidden_layer, state_is_tuple = False)
		self.W = tf.Variable(tf.random_normal([self.hidden_layer, self.vocab_size], stddev = 0.1))
		self.b = tf.Variable(tf.random_normal([self.vocab_size], stddev = 0.1))
		
#		self.train_vars.extend([self.emb_matrix, self.W, self.b])
		
#		with tf.variable_scope("LSTM") as vs :
#			_, _ = self.rnn_cell(tf.zeros([self.batch_size, self.emb_dim]), self.rnn_cell.zero_state(self.batch_size, 'float32'))
#			self.train_vars.extend([v for v in tf.all_variables() if v.name.startswith(vs.name)])
		self.lstm_out = {}
		self.lstm_prob = {}
		
		for i in range(self.seq_len + 1) :
			helper = LSTMHelper(self.batch_size, self.emb_matrix, self.seq_len, i, self.given_seq, self.start_ids, self.W, self.b)
			decoder = seq2seq.BasicDecoder(cell = self.rnn_cell, helper = helper, initial_state = self.rnn_cell.zero_state(self.batch_size, 'float32'))
			
			lstm_out, _, _ = seq2seq.dynamic_decode(decoder = decoder, maximum_iterations = self.seq_len)
			self.lstm_out[i] = lstm_out.sample_id
			# lstm_out.sample_id : batch_size * seq_len
			
			rnn_output = tf.matmul(tf.reshape(lstm_out.rnn_output, [-1, self.hidden_layer]), self.W) + self.b
			self.lstm_prob[i] = tf.reshape(rnn_output, [self.batch_size, self.seq_len, self.vocab_size])
			# lstm_out.rnn_output : batch_size * seq_len * hidden_layer
		
		self.pretrain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.lstm_prob[self.seq_len], labels = self.given_seq))
		self.pretrain = tf.contrib.estimator.clip_gradients_by_norm(
			tf.train.AdamOptimizer(learning_rate = self.pre_learning_rate), clip_norm = self.grad_clip).minimize(self.pretrain_loss)
		
		self.log_prob = tf.log(tf.clip_by_value(self.lstm_prob[self.seq_len], 1e-8, 1.0 - 1e-8))
#		self.train_loss = -tf.reduce_mean(tf.squeeze(tf.batch_gather(self.log_prob, tf.expand_dims(self.lstm_out[self.seq_len], -1))) * self.rewards)
		self.train_loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(self.lstm_out[seq_len], vocab_size) * self.log_prob, -1) * self.rewards)
		self.train = tf.contrib.estimator.clip_gradients_by_norm(
			tf.train.AdamOptimizer(learning_rate = self.adv_learning_rate), clip_norm = self.grad_clip).minimize(self.train_loss)
	
###################################################################################################
	
	def pretrain_batch(self, data) :
		feed = {self.given_seq: data}
		loss, _ = self.sess.run([self.pretrain_loss, self.pretrain], feed_dict = feed)
		return loss
		
	def evaluate_batch(self, data) :
		feed = {self.given_seq: data}
		loss, _ = self.sess.run([self.pretrain_loss, self.pretrain], feed_dict = feed)
		return loss
	
	def generate(self, given_len = 0, given_seq = None) :
		if given_len == 0 :
			given_seq = np.zeros([self.batch_size, self.seq_len])
		feed = {self.given_seq: given_seq}
		return self.sess.run(self.lstm_out[given_len], feed_dict = feed)
	
	def train_batch(self, discriminator) :
		samples = self.generate()
		rewards = np.zeros([self.batch_size, self.seq_len])
		
		if self.montecarlo_k is not None :
			idx_list = list(range(0, self.seq_len, self.montecarlo_k))
		else :
			idx_list = list(range(0, self.seq_len))
			
		for i in idx_list :
			if self.montecarlo_p is not None and random.random() > self.montecarlo_p :
				continue
			for j in range(self.montecarlo_num) :
				montecarlo_samples = self.generate(given_len = i, given_seq = samples)
				feedback = discriminator.feedback(montecarlo_samples)
				rewards[:, i] += feedback
				
		rewards /= self.montecarlo_num
#		print(rewards)
		
		feed = {self.given_seq: samples, self.rewards: rewards}
		loss, _ = self.sess.run([self.train_loss, self.train], feed_dict = feed)
		
		return loss
