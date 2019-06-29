__author__ = 'Nick Hirakawa, Verena Pongratz'

from .invdx import build_data_structures
from .rank import score_BM25, score_TFIDF
from .lm import score_lm_kld
from .bert import word_vecs_to_document_vec, score_cosine, string_to_bert_embeddings
from .specparser import SpecParserBert
from collections import defaultdict
import operator


class QueryProcessor:
	def __init__(self, queries, corpus, score_function="bm25"):
		self.queries = queries
		self.corpus = corpus
		self.index, self.dlt, self.dtf, self.tdf = build_data_structures(corpus)
		if score_function not in ("bm25", "tfidf", "lm", "bert"):
			print(f"ERROR: Unknown score function {score_function}! Using BM25.")
			score_function = "bm25"
		self.score_function = score_function

	def run(self):
		results = []
		for query in self.queries:
			results.append(self.run_query(query))
		return results

	def run_query(self, query):
		if self.score_function == "lm":
			return self.run_query_lm(query)
		elif self.score_function == "bert":
			return self.run_query_bert(query)
		query_result = dict()
		for term in query:
			if term in self.index:
				doc_dict = self.index[term]  # retrieve index entry
				for docid, freq in doc_dict.items():  # for each document and its word frequency
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

	def run_query_lm(self, query):
		query_result = dict()

		# Create query term frequency distribution
		query_term_fd = defaultdict(lambda: 1)
		for term in query:
			query_term_fd[term] += 1

		# Go over all documents that contain a term from the query
		seen_docs = set()
		for term in query:
			if term in self.index:
				doc_dict = self.index[term] # retrieve index entry
				for docid in doc_dict:
					if docid not in seen_docs:
						seen_docs.add(docid)
						query_result[docid] = score_lm_kld(query_term_fd, self.tdf[docid])

		# Calculate score as lowest KL-Divergence between query and document
		return query_result

	def run_query_bert(self, query):
		query_result = dict()

		# Create vector for query
		token_emb_pairs = string_to_bert_embeddings(query)
		query_vec = word_vecs_to_document_vec(tuple(zip(*token_emb_pairs))[1])

		# Go over all documents
		for docid, doc_vec in self.corpus.items():
			query_result[docid] = score_cosine(query_vec, doc_vec)

		return query_result


