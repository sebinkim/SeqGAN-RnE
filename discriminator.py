import numpy as np
import tensorflow as tf

class Discriminator :
	def __init__(self, sess, batch_size, seq_len, vocab_size, emb_dim, filters, grad_clip, l2_lambda, learning_rate) :
		
		self.sess = sess
		
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.grad_clip = grad_clip
		self.l2_lambda = l2_lambda
		self.learning_rate = learning_rate
		self.train_vars = []
		
		self.X = tf.placeholder(tf.int32, [None, self.seq_len])
		self.y = tf.placeholder(tf.int64, [None])
		self.dropout_rate = tf.placeholder(tf.float32)
		
		self.emb_matrix = tf.Variable(tf.random_normal([self.vocab_size, self.emb_dim], stddev = 0.1))
		self.emb_X = tf.expand_dims(tf.nn.embedding_lookup(self.emb_matrix, self.X), -1)
		# emb_X : batch_size * 2 x seq_len x emb_dim x 1

#		self.emb_sample = tf.expand_dims(tf.nn.embedding_lookup(self.emb_matrix, self.generator.sample), -1)
		# emb_sample : batch_size x seq_len x emb_dim x 1
		
		total_filters = sum([filter_num for _, filter_num in filters])
		self.W = tf.Variable(tf.random_normal([total_filters, 2], stddev = 0.1))
		self.b = tf.Variable(tf.random_normal([2], stddev = 0.1))
		self.l2_loss = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
		
#		self.train_vars.extend([self.emb_matrix, self.W, self.b])
		
		conv_out = []
		
		for filter_size, filter_num in filters :
			conv_W = tf.Variable(tf.random_normal([filter_size, self.emb_dim, 1, filter_num], stddev = 0.1))
			conv_b = tf.Variable(tf.random_normal([filter_num], stddev = 0.1))
			
#			self.train_vars.extend([conv_W, conv_b])
			
			H = tf.nn.conv2d(self.emb_X, conv_W, strides = [1, 1, 1, 1], padding = 'VALID')
			# H : (batch_size * 2) x (seq_len - filter_size + 1) x 1 x filter_num
			H = tf.nn.relu(tf.nn.bias_add(H, conv_b))
			H = tf.nn.max_pool(H, ksize = [1, self.seq_len - filter_size + 1, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID')
			# H : (batch_size * 2) x 1 X 1 x filter_num
			
			conv_out.append(H)
		
		self.conv_out = tf.nn.dropout(tf.reshape(tf.concat(conv_out, 3), [-1, total_filters]), rate = self.dropout_rate)
		# conv_out : (batch_size * 2) x total_filters
		
		self.output = tf.nn.softmax(tf.matmul(self.conv_out, self.W) + self.b)
		self.prob = tf.nn.softmax(self.output)
		
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.output, labels = self.y)) + self.l2_lambda * self.l2_loss
		self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output, 1), self.y), tf.float32))
		
		self.train = tf.contrib.estimator.clip_gradients_by_norm(
			tf.train.AdamOptimizer(learning_rate = self.learning_rate), clip_norm = self.grad_clip).minimize(self.loss)
	
	def pretrain_batch(self, batch_X, batch_y, dropout_rate) :
		feed = {self.X: batch_X, self.y: batch_y, self.dropout_rate: dropout_rate}
		loss, acc, _ = self.sess.run([self.loss, self.acc, self.train], feed_dict = feed)
		return loss, acc
	
	def train_batch(self, batch_X, batch_y, dropout_rate) :
		feed = {self.X: batch_X, self.y: batch_y, self.dropout_rate: dropout_rate}
		loss, acc, _ = self.sess.run([self.loss, self.acc, self.train], feed_dict = feed)
		return loss, acc
	
	def feedback(self, samples) :
		feed = {self.X: samples, self.dropout_rate: 0.0}
		feedback = self.sess.run(self.prob, feed_dict = feed)
		return feedback[:, 1]
