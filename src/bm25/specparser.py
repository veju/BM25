__author__ = 'Verena Pongratz'

import re
import codecs
import spacy
import json

from os.path import splitext, exists

from .cistem import stem

nlp = spacy.load("de")


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
		print("Read", len(self.corpus), "docs:")
		total_num_tokens = 0
		for chapter_name, chapter_tokens in self.corpus.items():
			total_num_tokens += len(chapter_tokens)
			print("  *", chapter_name, "(", len(chapter_tokens), "tokens)")
		avg_num_tokens = float(total_num_tokens)/float(len(self.corpus))
		print("Average token count per document:", avg_num_tokens)

	def get_corpus(self):
		return self.corpus

	def create_json_from_corpus(self):
		with codecs.open(self.json_filename, 'w') as json_output:
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
						tokens = [token.lemma_ for token in nlp(line)]
						print(tokens)
						current_chapter_tokens += tokens
				if current_chapter_tokens:
					self.corpus[current_chapter] = current_chapter_tokens
			print("Writing to", self.json_filename)
			json.dump(self.corpus, json_output, indent=2)

