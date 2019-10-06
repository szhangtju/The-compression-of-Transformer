import os 
from collections import Counteer, OrderedDict

import torch

class Vocab(object):
	def __init__(self, special=[], min_freq = 0, max_size = None, lower_case = True, delimiter=None, vocab_file=None):
		self.counter = Counter()
		self.special = special
		self.min_freq = min_freq
		self.max_size = max_size
		self.lower_case = lower_case
		self.delimiter = delimiter
		self.vocab_file = vocab_file
	def tokenize(self, line, add_eos=False, add_double_eos = False):
		line = line.strip()

		if self.lower_case:
			line = line.lower()

		if self.delimiter == '':
			symbols = line
		else:
			symbols = line.split(self.delimiter)
		if add_double_eos:
			return ['<S>'] +
