import os
import argparse
import shutil
import datetime
import pickle
import csv
import numpy as np
import tensorflow as tf
import random
import nltk.translate.bleu_score as bleu

from generator import Generator
from discriminator import Discriminator
from dataloder import Dataloader
from preprocess import *

gen_pretraining_epochs = 200
dis_pretraining_epochs = 50
training_epochs = 200
gen_training_epochs = 1 
dis_training_epochs = 1
dis_batch_epochs = 3
batch_size = 32
seq_len = 64
emb_dim = 32
hidden_layer = 128
montecarlo_num = 16
montecarlo_k = 1
montecarlo_p = None#0.8
filters = [[4, 64]] #, [8, 64], [12, 64], [16, 64]]
dis_dropout_rate = 0.2
dis_l2_lambda = 0.2
grad_clip = 5.0
gen_pre_learning_rate = 0.001
gen_adv_learning_rate = 0.001
dis_learning_rate = 0.00001

def evaluate_bleu(dataloader, generator, eval_num = 32, mode = 'sentence') :
	
	ref_list = []
	gen_list = []
	dataloader.test_init()

	if mode == 'corpus' :
		while not dataloader.test_finished() :
			batch = dataloader.test_next_batch()
			ref_list.extend(batch.astype(np.str).tolist())
			gen = generator.generate()
			gen_list.extend(gen.astype(np.str).tolist())
	
		return bleu.corpus_bleu(ref_list, gen_list,
			smoothing_function = bleu.SmoothingFunction().method4)
	else :
		while not dataloader.test_finished() :
			batch = dataloader.test_next_batch()
			ref_list.extend(batch.astype(np.str).tolist())
		
		for i in range(eval_num // batch_size) :
			gen = generator.generate()
			gen_list.extend(gen.astype(np.str).tolist())
		
		score8, score12 = 0., 0.
		for i, gen in enumerate(gen_list) :
			x8 = bleu.sentence_bleu(ref_list, gen, weights = (1.0 / 8,) * 8,
				smoothing_function = bleu.SmoothingFunction().method4)
			x12 = bleu.sentence_bleu(ref_list, gen, weights = (1.0 / 12,) * 12,
				smoothing_function = bleu.SmoothingFunction().method4)
			score8 += x8
			score12 += x12
		
		return score8 / len(gen_list), score12 / len(gen_list)

eval_step = 0

def evaluate_mle(dataloader, generator) :

	dataloader.test_init()
	total_batch = 0
	loss = 0.
	
	while not dataloader.test_finished() :
		batch = dataloader.test_next_batch()
		loss += generator.evaluate_batch(batch)
		total_batch = total_batch + 1
	
	return loss / total_batch	

def main() :
	parser = argparse.ArgumentParser()
	parser.add_argument('--pretrained', type = str, default = '')
	args = parser.parse_args()
	pretrained = args.pretrained
	
	dataloader = Dataloader('training_data', 'test_data', seq_len, batch_size)
	vocab_size = dataloader.vocab_size
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	
	if not os.path.isdir("results") :
		os.makedirs('results')
	if not os.path.isdir("logs") :
		os.makedirs('logs')
	
	nowdate = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	os.makedirs('results/' + nowdate)
	os.makedirs('results/' + nowdate + '/generated')
	
	hyp_file = open('results/' + nowdate + '/hyperparameters.txt', 'w')
	hyp_file.write('gen_pretraining_epochs = ' + str(gen_pretraining_epochs) + '\n')
	hyp_file.write('dis_pretraining_epochs = ' + str(dis_pretraining_epochs) + '\n')
	hyp_file.write('training_epochs = ' + str(training_epochs) + '\n')
	hyp_file.write('gen_training_epochs = ' + str(gen_training_epochs) + '\n')
	hyp_file.write('dis_training_epochs = ' + str(dis_training_epochs) + '\n')
	hyp_file.write('dis_batch_epochs = ' + str(dis_batch_epochs) + '\n')
	hyp_file.write('batch_size = ' + str(batch_size) + '\n')
	hyp_file.write('seq_len = ' + str(seq_len) + '\n')
	hyp_file.write('emb_dim = ' + str(emb_dim) + '\n')
	hyp_file.write('hidden_layer = ' + str(hidden_layer) + '\n')
	hyp_file.write('montecarlo_num = ' + str(montecarlo_num) + '\n')
	hyp_file.write('montecarlo_k = ' + str(montecarlo_k) + '\n')
	hyp_file.write('filters = ' + str(filters) + '\n')
	hyp_file.write('dis_dropout_rate = ' + str(dis_dropout_rate) + '\n')
	hyp_file.write('dis_l2_lambda = ' + str(dis_l2_lambda) + '\n')
	hyp_file.write('grad_clip = ' + str(grad_clip) + '\n')
	hyp_file.write('gen_pre_learning_rate = ' + str(gen_pre_learning_rate) + '\n')
	hyp_file.write('gen_adv_learning_rate = ' + str(gen_adv_learning_rate) + '\n')
	hyp_file.write('dis_learning_rate = ' + str(dis_learning_rate) + '\n')
	hyp_file.close()
	
	csv_file = open('results/' + nowdate + '/train_result.csv', 'w', newline = '')
	csv_writer = csv.DictWriter(csv_file, fieldnames =
		['time', 'pre_gen_loss', 'pre_dis_loss', 'pre_dis_acc',
		'adv_gen_loss', 'adv_dis_loss', 'adv_dis_acc', 'mle', 'bleu'])
	csv_writer.writeheader()
	
#	tf.reset_default_graph()

	with tf.Session(config = config) as sess :

		generator = Generator(sess, batch_size, seq_len, vocab_size, emb_dim, hidden_layer,
			montecarlo_num, grad_clip, gen_pre_learning_rate, gen_adv_learning_rate, montecarlo_k = montecarlo_k, montecarlo_p = montecarlo_p)
		discriminator = Discriminator(sess, batch_size, seq_len, vocab_size, emb_dim, filters,
			grad_clip, dis_l2_lambda, dis_learning_rate)
		
		writer = tf.summary.FileWriter('./logs/', sess.graph)
		
		pre_gen_summary = tf.Summary()
		pre_gen_summary.value.add(tag = 'pre_gen_loss', simple_value = None)
		pre_dis_summary = tf.Summary()
		pre_dis_summary.value.add(tag = 'pre_dis_loss', simple_value = None)
		pre_dis_summary.value.add(tag = 'pre_dis_acc', simple_value = None)
		
		adv_summary = tf.Summary()
		adv_summary.value.add(tag = 'adv_gen_loss', simple_value = None)
		adv_summary.value.add(tag = 'adv_dis_loss', simple_value = None)
		adv_summary.value.add(tag = 'adv_dis_acc', simple_value = None)
		
		mle_summary = tf.Summary()
		mle_summary.value.add(tag = 'mle', simple_value = None)
			
		exc_time = datetime.timedelta(0)
		
		if pretrained != '' :
			print('Restoring\n')
#			save_filename = pretrained
#			saver = tf.train.import_meta_graph(save_filename + '.meta')
#			saver.restore(sess, save_filename)
#			tf.train.Saver().restore(sess, save_filename)
			
		else :
			sess.run(tf.global_variables_initializer())
			
			print('Pretraining Generator\n')
			
			for epoch in range(gen_pretraining_epochs) :
				dataloader.train_init()
				total_batch = 0
				loss = 0.
				while not dataloader.train_finished() :
					batch = dataloader.train_next_batch()
					t = datetime.datetime.now()
					loss += generator.pretrain_batch(batch)
					exc_time += datetime.datetime.now() - t
					total_batch = total_batch + 1
				
				loss /= total_batch
				print('Epoch %3d: %.6f\n' % (epoch, loss))
				pre_gen_summary.value[0].simple_value = loss
				writer.add_summary(pre_gen_summary, epoch)
				
				eval_mle = evaluate_mle(dataloader, generator)
				print('MLE : %.6f\n' % eval_mle)
				mle_summary.value[0].simple_value = eval_mle
				writer.add_summary(mle_summary, epoch)
				
				csv_writer.writerow({'time': exc_time.seconds, 'pre_gen_loss': loss, 'mle': eval_mle})
				
#			print('BLEU score : %.6f %.6f\n' % evaluate_bleu(dataloader, generator))
			
#			print('Generating\n')
#			
#			gen_melody = generator.generate()
#			
#			for i, melody in enumerate(gen_melody) :
#				gen_filename = 'results/' + nowdate + '/generated/' + pre-' + str(i) + '.mid'
#				print('Generating ' + gen_filename[10:-4] + '...')
#				print(melody)
#				dataloader.write_melody(gen_filename, melody)
			
			print('Pretraining Discrimininator\n')
			
			for epoch in range(dis_pretraining_epochs) :
				dataloader.train_init()
				total_batch = 0
				loss, acc = 0., 0.
				while not dataloader.train_finished() :
					real_batch = dataloader.train_next_batch()
					fake_batch = generator.generate()
					batch_X = np.concatenate([real_batch, fake_batch])
					batch_y = np.concatenate([np.ones((batch_size)), np.zeros((batch_size))])
					perm = np.random.permutation(batch_size * 2)
					
					batch_loss, batch_acc = 0., 0.
					for _ in range(dis_batch_epochs) :
						t = datetime.datetime.now()
						_loss, _acc = discriminator.pretrain_batch(batch_X[perm], batch_y[perm], dis_dropout_rate)
						exc_time += datetime.datetime.now() - t
						batch_loss += _loss
						batch_acc += _acc
						
					loss += batch_loss / dis_batch_epochs
					acc += batch_acc / dis_batch_epochs
					
					total_batch = total_batch + 1
				
				loss /= total_batch
				acc /= total_batch
				print('Epoch %3d: %.6f, Accuracy: %.6f\n' % (epoch, loss, acc))
				pre_dis_summary.value[0].simple_value = loss
				pre_dis_summary.value[1].simple_value = acc
				writer.add_summary(pre_dis_summary, epoch)
				
				csv_writer.writerow({'time': exc_time.seconds, 'pre_dis_loss': loss, 'pre_dis_acc': acc})
			
			save_filename = 'results/' + nowdate + '/save-pre.ckpt'
			tf.train.Saver().save(sess, save_filename)
			print('Saved at ' + save_filename + '\n')
			
#		print('BLEU score : %.6f %.6f\n' % evaluate_bleu(dataloader, generator, eval_num = 256))
		
		print('Adversarial Training\n')
		
		dataloader.train_init()
		
		for epoch in range(training_epochs) :
			gen_loss = 0.
			for gen_epoch in range(gen_training_epochs) :
				t = datetime.datetime.now()
				loss = generator.train_batch(discriminator)
				exc_time += datetime.datetime.now() - t
				gen_loss += loss
					
				print('Epoch %3d - %3d: %.6f' % (epoch, gen_epoch, loss))
			
			dis_loss, dis_acc = 0., 0.
			for dis_epoch in range(dis_training_epochs) :
				dataloader.train_init()
				total_batch = 0
				loss, acc = 0., 0.
				while not dataloader.train_finished() :
					real_batch = dataloader.train_next_batch()
					fake_batch = generator.generate()
					batch_X = np.concatenate([real_batch, fake_batch])
					batch_y = np.concatenate([np.ones((batch_size)), np.zeros((batch_size))])
					perm = np.random.permutation(batch_size * 2)
					
					batch_loss, batch_acc = 0., 0.
					for _ in range(dis_batch_epochs) :
						t = datetime.datetime.now()
						_loss, _acc = discriminator.train_batch(batch_X[perm], batch_y[perm], dis_dropout_rate)
						exc_time += datetime.datetime.now() - t
						batch_loss += _loss
						batch_acc += _acc
						
					loss += batch_loss / dis_batch_epochs
					acc += batch_acc / dis_batch_epochs
					
					total_batch = total_batch + 1
				
				dis_loss += loss / total_batch
				dis_acc += acc / total_batch
					
				print('Epoch %3d - %3d: %.6f, Accuracy: %.6f' % (epoch, dis_epoch, loss / total_batch, acc / total_batch))
			
			gen_loss /= gen_training_epochs
			dis_loss /= dis_training_epochs
			dis_acc /= dis_training_epochs
			print('Epoch %3d: %.6f / %.6f, %.6f\n' % (epoch, gen_loss, dis_loss, dis_acc))
			adv_summary.value[0].simple_value = gen_loss
			adv_summary.value[1].simple_value = dis_loss
			adv_summary.value[2].simple_value = dis_acc
			writer.add_summary(adv_summary, epoch)
				
			eval_mle = evaluate_mle(dataloader, generator)
			print('MLE : %.6f\n' % eval_mle)
			mle_summary.value[0].simple_value = eval_mle
			writer.add_summary(mle_summary, gen_pretraining_epochs + epoch)
			
			csv_writer.writerow({'time': exc_time.seconds, 'adv_gen_loss': gen_loss,
				'adv_dis_loss': dis_loss, 'adv_dis_acc': dis_acc, 'mle': eval_mle})
			
#			if epoch % 10 == 9 :
#				print('BLEU score : %.6f %.6f\n' % evaluate_bleu(dataloader, generator))
			
			print('Generating\n')
			
			gen_melody = generator.generate()
			
			for i, melody in enumerate(gen_melody) :
				gen_filename = 'results/' + nowdate + '/generated/' + str(epoch) + '-' + str(i) + '.mid'
				dataloader.write_melody(gen_filename, melody)
		
		eval_bleu = evaluate_bleu(dataloader, generator, eval_num = 64)
		print('BLEU score : %.6f %.6f\n' % eval_bleu)
		
		csv_writer.writerow({'bleu': eval_bleu})

		save_filename = 'results/' + nowdate + '/save-adv.ckpt'
		tf.train.Saver().save(sess, save_filename)
		
		csv_file.close()

if __name__ == '__main__' :
	main()
