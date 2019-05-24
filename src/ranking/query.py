__author__ = 'Nick Hirakawa, Verena Pongratz'

from .invdx import build_data_structures
from .rank import score_BM25, score_TFIDF
import operator

class QueryProcessor:
	def __init__(self, queries, corpus, score_function="bm25"):
		self.queries = queries
		self.corpus = corpus
		self.index, self.dlt, self.dtf = build_data_structures(corpus)
		if score_function not in ("bm25", "tfidf"):
			print(f"ERROR: Unknown score function {score_function}! Using BM25.")
			score_function = "bm25"
		self.score_function = score_function

	def run(self):
		results = []
		for query in self.queries:
			results.append(self.run_query(query))
		return results

	def run_query(self, query):
		query_result = dict()
		for term in query:
			if term in self.index:
				doc_dict = self.index[term] # retrieve index entry
				for docid, freq in doc_dict.items(): #for each document and its word frequency
					if self.score_function == "bm25":
						score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
					 					   dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length()) # calculate score
					elif self.score_function == "tfidf":
						score = score_TFIDF(f=freq, n=len(self.corpus), dtf=self.dtf[term])

					if docid in query_result: #this document has already been scored once
						query_result[docid] += score
					else:
						query_result[docid] = score
			else:
				print("Term", term, "not present in index!")
		return query_result