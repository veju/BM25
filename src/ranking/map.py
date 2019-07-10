import codecs
from collections import defaultdict
from typing import Dict, List, Tuple

"""
Returns TSV file content indexed by first column
"""
def parse_tsv(file_path) -> Dict[int, List[Tuple[str]]]:
    print(f"Parsing {file_path} ...")
    result = defaultdict(list)
    with codecs.open(file_path, 'r') as file:
        for line in file:
            if '\t' in line:
                data = tuple(word.strip() for word in line.split('\t') if len(word.strip()))
            else:
                data = tuple(word.strip() for word in line.split('    ') if len(word.strip()))
            result[int(data[0])].append(data)
    print(f"  ... found {len(result)} queries.")
    return result

"""
Calculate MAP@n score for gold+ranking combination.
Params:
- n: 1-n Precision values to calculate per document
- gold_file_path: Path to TSV file with (qid, index, docid, score, model, gold_place)
- ranking_file_path: Path to TSV file with (qid, index, docid, score, model)
"""
def score_map(n, gold_file_path, ranking_file_path):
    gold_data = parse_tsv(gold_file_path)
    ranking_data = parse_tsv(ranking_file_path)
    map_score = 0

    for query_id, gold_ranking in gold_data.items():
        # Find top 5 gold documents in gold ranking
        gold_doc_per_place = dict()
        for document_data in gold_ranking:
            if len(document_data) > 5:
                gold_place = int(document_data[5])
                if gold_place in gold_doc_per_place:
                    print(f"Found multiple docs for query {query_id}, place {gold_place}!")
                else:
                    gold_doc_per_place[gold_place] = document_data[2]
        if len(gold_doc_per_place) < n:
            print(f"Found only {len(gold_doc_per_place)} relevance judgements for query {query_id}")
        # Calculate avg. precision for top 1-n documents
        avg_prec = .0
        # Go over multiple top-n
        for top_n in range(n):
            prec = 0
            # Go over all gold documents for top-n
            for place in range(top_n+1):
                if place+1 not in gold_doc_per_place:
                    print(f"No gold doc for place {place+1} in query {query_id}!")
                    continue
                gold_doc_id = gold_doc_per_place[place+1]
                # Check whether gold document is present in top-n ranked docs
                for ranked_doc_data in ranking_data[query_id][:top_n+1]:
                    if ranked_doc_data[2] == gold_doc_id:
                        prec += 1./(top_n+1.)
                        break
            avg_prec += prec
        avg_prec /= float(n)
        map_score += avg_prec

    # avg. precision over all queries
    map_score /= len(gold_data)
    return map_score
