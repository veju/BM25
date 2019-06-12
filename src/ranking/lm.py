__author__ = 'Verena Pongratz'

from typing import Dict, Set
from math import log


def create_prob_dist(term_freq: Dict[str, int], vocab_set: Set[str]):
    # Term frequency dictionaries are expected to be defaultdict(lambda: 1)
    total_term_count = sum(term_freq[term] for term in vocab_set)
    result = {term: float(term_freq[term])/float(total_term_count) for term in vocab_set}
    return result


def score_lm_kld(query_term_freq: Dict[str, int], document_term_freq: Dict[str, int]):
    # Term frequency dictionaries are expected to be defaultdict(lambda: 1)
    # Determine combined token set
    all_terms = set(query_term_freq) | set(document_term_freq)
    # Create query & doc probability distributions from token set
    term_pd_query = create_prob_dist(query_term_freq, all_terms)
    term_pd_document = create_prob_dist(document_term_freq, all_terms)
    # Calculate KL divergence between pds
    kld = sum(
        term_pd_query[term] *
        log(term_pd_query[term]/term_pd_document[term])
        for term in all_terms)
    return kld

