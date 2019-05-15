__author__ = 'Verena Pongratz'

import re
import codecs


class SpecParser:

	def __init__(self, filename):
		self.filename = filename
		self.regex = re.compile(r"^[ \t]*([0-9]\.)*[0-9]+[ \t]+[A-Za-z][^\t\n]*$")
		self.corpus = dict()

	def parse(self):
		with codecs.open(self.filename) as f:
			current_chapter = ""
			current_chapter_tokens = []
			for line in f.readlines():
				line = line.rstrip() # get rid of newline
				if self.regex.match(line) is not None:
					if current_chapter_tokens and current_chapter:
						self.corpus[current_chapter] = current_chapter_tokens
					current_chapter = line.strip()
					current_chapter_tokens = []
				else:
					current_chapter_tokens += line.split()
			if current_chapter_tokens:
				self.corpus[current_chapter] = current_chapter_tokens
		print("Found", len(self.corpus), "docs:")
		total_num_tokens = 0
		for chapter_name, chapter_tokens in self.corpus.items():
			total_num_tokens += len(chapter_tokens)
			print("  *", chapter_name, "(", len(chapter_tokens), "tokens)")
		avg_num_tokens = float(total_num_tokens)/float(len(self.corpus))
		print("Average token count per document:", avg_num_tokens)

	def get_corpus(self):
		return self.corpus