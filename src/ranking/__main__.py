__author__ = 'Nick Hirakawa, Verena Pongratz'


from .parse import *
from .query import QueryProcessor
from .specparser import SpecParser, SpecParserBert
import operator
import sys

score_function = "bm25"
if len(sys.argv) > 1:
	score_function = sys.argv[1]


def main():
	global score_function
	qp = QueryParser(filename='text/queries.txt')
	if score_function == "bert":
		cp = SpecParserBert(filename='text/corpus.txt')
	else:
		cp = SpecParser(filename='text/corpus.txt')
	qp.parse()
	queries = qp.get_queries()
	cp.parse()
	corpus = cp.get_corpus()
	proc = QueryProcessor(queries, corpus, score_function)
	results = proc.run()
	qid = 0
	for result in results:
		sorted_x = sorted(iter(result.items()), key=operator.itemgetter(1))
		sorted_x.reverse()
		index = 0
		for i in sorted_x[:100]:
			tmp = (qid, index, i[0], str(i[1])[:5], score_function)
			print('{:>1}\t{:>4}\t{:>2}\t{:>12}\t{}'.format(*tmp))
			index += 1
		qid += 1


if __name__ == '__main__':
	main()