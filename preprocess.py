import os
import random
import numpy as np
import music21
import fileinput
import pickle
from sklearn.preprocessing import LabelEncoder

def read_data(data_filepath) :
	num_adj = ["st", "nd", "rd", "th"]

	melody_array = []
	
	info_filename = data_filepath + 'nottingham_info.txt'
	cnt = 0
	
	song_list = list(fileinput.input([info_filename]))
	random.shuffle(song_list)
	
	train_num = int(len(song_list) * 0.8)

	for i, line in enumerate(song_list) :
		filename, main_inst = line.split(',')
		print('Encoding', str(cnt + 1) + num_adj[min(cnt % 10, 3)], 'Song,', filename + '!')
		
		filename = data_filepath + filename + '.mid'
		main_melody, length = read_midi(filename, int(main_inst))
		
		if i < train_num :
			save_filename = 'training_data/#' + str(cnt)
			with open(save_filename, 'wb') as fout :
				pickle.dump(main_melody, fout)
			print('length :', str(length) + ', saved in', save_filename, '\n')
		else :
			save_filename = 'test_data/#' + str(cnt)
			with open(save_filename, 'wb') as fout :
				pickle.dump(main_melody, fout)
			print('length :', str(length) + ', saved in', save_filename, '\n')
		
		cnt += 1
		
#			write_midi(save_filename[:-3] + 'mid', decode_melody(main_melody)) ## testing


def read_midi(midi_filename, main_inst) :
	mf = music21.midi.MidiFile()
	mf.open(midi_filename)
	mf.read()
	full_midi = music21.midi.translate.midiFileToStream(mf)
	mf.close()
	
	main_melody = full_midi.parts[main_inst].flat
	
	return encode_melody(main_melody)


def encode_melody(melody) :
	x = music21.interval.Interval(melody.analyze('key').tonic, music21.pitch.Pitch('C4'))
	melody = melody.transpose(x)
	
	melody_length = int(melody.highestTime * 4 + 0.1)
	encoded_melody = np.zeros(melody_length)
	
	for note in melody.getElementsByClass(music21.note.Note) :
		pt = note.pitch.midi
		
		st = int(note.offset * 4 + 0.1)
		en = int((note.offset + note.quarterLength) * 4 + 0.1)
		
		encoded_melody[st] = max(encoded_melody[st], pt * 2)
		for i in range(st + 1, en) :
			encoded_melody[i] = max(encoded_melody[i], pt * 2 + 1)
	
	for chord in melody.getElementsByClass(music21.chord.Chord) :
		pt = max(chord, key=lambda n: n.pitch.midi).pitch.midi
		
		st = int(chord.offset * 4 + 0.1)
		en = int((chord.offset + chord.quarterLength) * 4 + 0.1)
		
		encoded_melody[st] = max(encoded_melody[st], pt * 2)
		for i in range(st + 1, en) :
			encoded_melody[i] = max(encoded_melody[i], pt * 2 + 1)
			
	encoded_melody = np.trim_zeros(encoded_melody)

	shortened_melody = []
#	shortened_melody = ['<SOS>'] # start of sequence
	
	for i in range(len(encoded_melody)):
		if i > 0 and int(encoded_melody[i] + 0.1) == 0 and int(encoded_melody[i - 1] + 0.1) == 0 : continue
		
		if int(encoded_melody[i]) == 0 :
			shortened_melody.append('0')
		elif i > 0 and int(encoded_melody[i]) % 2 == 1 :
			shortened_melody.append(str(int(encoded_melody[i] + 0.1) // 2) + '+')
		else :
			shortened_melody.append(str(int(encoded_melody[i] + 0.1) // 2))

#	shortened_melody.append('<EOS>') # end of sequence
	
	print(shortened_melody)

	return shortened_melody, len(shortened_melody)

def write_midi(midi_filename, midi_stream) :
	mf = music21.midi.translate.streamToMidiFile(midi_stream)
	mf.open(midi_filename, 'wb')
	mf.write()
	mf.close()


def decode_melody(melody) :
	melody_length = len(melody)
	decoded_melody = music21.stream.Stream()
	
	for i in range(melody_length) :
		if melody[i] == '<EOS>' : break
		elif melody[i][0] == '<' or melody[i] == '0' or melody[i][-1] == '+' : continue
		
		j = i + 1
		while j < melody_length and melody[j] == melody[i] + '+' :
			j += 1
		
		note = music21.note.Note()
		note.pitch.midi = int(melody[i])
		note.quarterLength = (j - i) / 4
		off = (i - 1) / 4
		
		decoded_melody.insert(off, note)
	
	return decoded_melody
