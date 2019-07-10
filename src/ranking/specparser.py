__author__ = 'Verena Pongratz'

import re
import codecs
import json

from os.path import splitext, exists
from .bert import word_vecs_to_document_vec, nlp, string_to_bert_embeddings
from numpy import asarray


class SpecParser:

	def __init__(self, filename):
		self.filename = filename
		self.json_filename = splitext(filename)[0]+".json"
		self.regex = re.compile(r"^[ \t]*([0-9]\.)*[0-9]+[ \t]+[A-Za-z][^\t\n]*$")
		self.corpus = dict()

	def parse(self):
		if not exists(self.json_filename):
			print("Could not find", self.json_filename, "creating...")
			self.create_json_from_corpus()
		else:
			print("Found", self.json_filename, "!")
			with codecs.open(self.json_filename) as file:
				self.corpus = json.load(file)
		print("Read", len(self.corpus), "docs.")
		total_num_tokens = 0
		for chapter_name, chapter_tokens in self.corpus.items():
			total_num_tokens += len(chapter_tokens)
		#	print("  *", chapter_name, "(", len(chapter_tokens), "tokens)")
		avg_num_tokens = float(total_num_tokens)/float(len(self.corpus))
		print("Average token count per document:", avg_num_tokens)

	def get_corpus(self):
		return self.corpus

	def tokenize(self, paragraph):
		return [token.lemma_.lower() for token in nlp(paragraph)]

	def after_create(self, corpus):
		with codecs.open(self.json_filename, 'w') as json_output:
			print("Writing to", self.json_filename)
			json.dump(self.corpus, json_output, indent=2)

	def create_json_from_corpus(self):
		with codecs.open(self.filename) as orig_corpus:
			current_chapter = ""
			current_chapter_tokens = []
			for line in orig_corpus.readlines():
				line = line.rstrip() # get rid of newline
				if self.regex.match(line) is not None:
					if current_chapter_tokens and current_chapter:
						self.corpus[current_chapter] = current_chapter_tokens
					current_chapter = line.strip()
					current_chapter_tokens = []
				else:
					tokens = self.tokenize(line)
					current_chapter_tokens += tokens
			if current_chapter_tokens:
				self.corpus[current_chapter] = current_chapter_tokens
			self.after_create(self.corpus)


class SpecParserBert(SpecParser):
	
	def __init__(self, filename):
		self.filename = filename
		self.json_filename = splitext(filename)[0]+".json"
		self.bert_json_filename = splitext(filename)[0]+".bert.json"
		self.regex = re.compile(r"^[ \t]*([0-9]\.)*[0-9]+[ \t]+[A-Za-z][^\t\n]*$")
		self.corpus = dict()
		self.doc_vec_corpus = dict()

	def parse(self):
		if exists(self.bert_json_filename):
			print("Found", self.bert_json_filename, "!")
			with codecs.open(self.bert_json_filename) as file:
				self.doc_vec_corpus = {
					docid: asarray(docvec)
					for docid, docvec in dict(json.load(file)).items()
				}
		else:
			print("Could not find", self.bert_json_filename, "creating...")
		super().parse()

	def get_corpus(self):
		return self.doc_vec_corpus

	# Generate (token, embedding)-pair list from paragraph
	def tokenize(self, paragraph):
		return string_to_bert_embeddings(paragraph)

	def after_create(self, corpus):
		print("Generating average BERT embeddings ...")
		for docid, word_and_embedding_list in corpus.items():
			words_and_embeddings = tuple(zip(*word_and_embedding_list))
			corpus[docid] = words_and_embeddings[0]
			self.doc_vec_corpus[docid] = word_vecs_to_document_vec(words_and_embeddings[1])
		with codecs.open(self.bert_json_filename, 'w') as json_output:
			print("Writing to", self.bert_json_filename)
			json.dump({  # convert numpy arrays to lists
				docid: list(docvec) for docid, docvec in self.doc_vec_corpus.items()
			}, json_output, indent=2)
		super().after_create(corpus)

