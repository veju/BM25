__author__ = 'Verena Pongratz'

import re


class SpecParser:

	def __init__(self, filename):
		self.filename = filename
		self.regex = re.compile(r"^[ \t]*([0-9]\.)*[0-9]+[ \t]+[A-Za-z][^\t\n]*$")
		self.corpus = dict()

	def parse(self):
		with open(self.filename) as f:
			current_chapter = ""
			current_chapter_tokens = []
			for line in f.readlines():
				line = line.rstrip() # get rid of newline
				if self.regex.match(line) is not None:
					if current_chapter_tokens and current_chapter:
						self.corpus[current_chapter] = current_chapter_tokens
						print current_chapter, current_chapter_tokens
					current_chapter = line.strip()
					current_chapter_tokens = []
				else:
					current_chapter_tokens += line.split()
			if current_chapter_tokens:
				self.corpus[current_chapter] = current_chapter_tokens

	def get_corpus(self):
		return self.corpus