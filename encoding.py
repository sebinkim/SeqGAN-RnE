import os
import shutil

from preprocess import *

shutil.rmtree('training_data')
os.makedirs('training_data')
shutil.rmtree('test_data')
os.makedirs('test_data')
read_data('nottingham/')
