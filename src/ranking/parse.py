__author__ = 'Nick Hirakawa, Verena Pongratz'

import re

import spacy
from .bert import nlp


class CorpusParser:

	def __init__(self, filename):
		self.filename = filename
		self.regex = re.compile('^#\s*\d+')
		self.corpus = dict()

	def parse(self):
		with open(self.filename) as f:
			s = ''.join(f.readlines())
		blobs = s.split('#')[1:]
		for x in blobs:
			text = x.split()
			docid = text.pop(0)
			self.corpus[docid] = text

	def get_corpus(self):
		return self.corpus


class QueryParserRaw:

	def __init__(self, filename):
		self.filename = filename
		self.queries = []

	def parse(self):
		with open(self.filename) as f:
			for line in f.readlines():
				self.queries.append(line)

	def get_queries(self):
		return self.queries


class QueryParser:

	def __init__(self, filename):
		self.filename = filename
		self.queries = []

	def parse(self):
		with open(self.filename) as f:
			for line in f.readlines():
				self.queries.append([token.lemma_ for token in nlp(line)])

	def get_queries(self):
		return self.queries
