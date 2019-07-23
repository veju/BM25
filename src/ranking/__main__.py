__author__ = 'Nick Hirakawa, Verena Pongratz'


from .parse import *
from .query import QueryProcessor
from .specparser import SpecParser, SpecParserBert
from .map import score_map
import operator
import sys
import argparse


parser = argparse.ArgumentParser(description='Run diverse IR scoring functions on diverse doc/question sets.')
parser.add_argument('--map', nargs=2, dest='map', help='Pass path to gold standard (TSV) and path to ranking (TSV) to calculate MAP@5 score.')
parser.add_argument('--model', dest='model', default='bm25', help='IR model to use', choices=['bm25', 'lm', 'tfidf', 'bert'])
parser.add_argument('--queries', dest='queries', help='Path to query file (One query per line)')
parser.add_argument('--docs', dest='docs', default='bm25', help="""
Path to document file. Will be parsed: Documents begin with heading acc. to
a regular expression: /^[ \t]*([0-9]\.)*[0-9]+[ \t]+[A-Za-z][^\t\n]*$/;
content under heading is adopted as document content.
""")
args = parser.parse_args()


def main():
    global args
    if args.map:
        map_score = score_map(5, args.map[0], args.map[1])
        print(f"Calculated MAP@5 score for {args.map[0]}, {args.map[1]}")
        for n, prec_at_n in enumerate(map_score[0]):
            print(f"Prec@{n+1}: {prec_at_n}")
        print(f"MAP@5 = {map_score[1]}")
        return

    if args.model == "bert":
        cp = SpecParserBert(filename=args.docs)
        qp = QueryParserRaw(filename=args.queries)
    else:
        cp = SpecParser(filename=args.docs)
        qp = QueryParser(filename=args.queries)

    qp.parse()
    queries = qp.get_queries()
    cp.parse()
    corpus = cp.get_corpus()

    proc = QueryProcessor(queries, corpus, args.model)
    results = proc.run()
    qid = 0
    for result in results:
        sorted_x = sorted(iter(result.items()), key=operator.itemgetter(1))
        sorted_x.reverse()
        index = 0
        for i in sorted_x[:100]:
            tmp = (qid, index, i[0], str(i[1])[:5], args.model)
            print('{:>1}\t{:>4}\t{:>2}\t{:>12}\t{}'.format(*tmp))
            index += 1
        qid += 1


if __name__ == '__main__':
    main()