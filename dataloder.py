import numpy as np
import random

from preprocess import *

class Dataloader :
	def __init__(self, training_filepath, test_filepath, seq_len, batch_size) : 
		
		self.seq_len = seq_len
		self.batch_size = batch_size

		self.word_to_int = {}
		self.int_to_word = {}
		self.vocab_size = 1
		
		self.word_to_int['<START>'] = 0
		self.int_to_word[0] = '<START>'
		
		self.training_data = []
		self.training_data = self.read(self, training_filepath)

		self.test_data = []
		self.test_data = self.read(self, test_filepath)
		
		print('training_data :', self.training_data.shape)
		print('test_data :', self.test_data.shape)
		print('vocab_size :', self.vocab_size)
	
	def read(self, dataset, filepath) :
		
		dataset = []
		data = os.listdir(filepath)
		random.shuffle(data)

		for filename in data:
			emb_melody = []
			with open(filepath + '/' + filename, 'rb') as fin :
				melody = pickle.load(fin)
				for word in melody :
					if word in self.word_to_int :
						emb_melody.append(self.word_to_int[word])
					else :
						self.word_to_int[word] = self.vocab_size
						self.int_to_word[self.vocab_size] = word
						emb_melody.append(self.vocab_size)
						self.vocab_size += 1
			
			dataset.extend(emb_melody)

		data_size = len(dataset) // (self.batch_size * self.seq_len)
		dataset = np.array(dataset[:data_size * self.batch_size * self.seq_len])
		dataset = np.reshape(dataset, (-1, self.batch_size, self.seq_len))
		
		return dataset
		
###################################################################################################
	
	def train_init(self) :
		np.random.shuffle(self.training_data)
		self.train_i = 0
		
	def train_finished(self) :
		return self.train_i >= len(self.training_data)
	
	def train_next_batch(self) :
		self.train_i = self.train_i + 1
		return self.training_data[self.train_i - 1]
		
###################################################################################################
	
	def test_init(self) :
		np.random.shuffle(self.test_data)
		self.test_i = 0
		
	def test_finished(self) :
		return self.test_i >= len(self.test_data)
	
	def test_next_batch(self) :
		self.test_i = self.test_i + 1
		return self.test_data[self.test_i - 1]
		
###################################################################################################
	
	def decode_melody(self, melody) :
		melody = [self.int_to_word[t] for t in melody]
		decoded_melody = music21.stream.Stream()
		
		for i in range(len(melody)) :
			if melody[i] == '<START>' or melody[i] == '0' or melody[i][-1] == '+' : continue
			
			j = i + 1
			while j < len(melody) and melody[j] == melody[i] + '+' :
				j += 1
			
			note = music21.note.Note()
			note.pitch.midi = int(melody[i])
			note.quarterLength = (j - i) / 4
			off = i / 4
			
			decoded_melody.insert(off, note)
		
		return decoded_melody
	
	def write_melody(self, midi_filename, melody) :
		midi_stream = self.decode_melody(melody)
		mf = music21.midi.translate.streamToMidiFile(midi_stream)
		mf.open(midi_filename, 'wb')
		mf.write()
		mf.close()
