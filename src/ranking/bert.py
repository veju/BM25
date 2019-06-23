import numpy as np
from typing import List, Tuple


def string_to_bert_embeddings(input: str) -> Tuple[List[np.array], List[str]]:
    pass


def score_cosine(qv: np.array, dv: np.array) -> float:
    pass


def word_vecs_to_document_vec(vecs: List[np.array], weights: np.array=None) -> np.array:
    assert(len(vecs) > 0)
    result = np.zeros(vecs[0].shape)
    for vec in vecs:
        # TODO: Consider weight
        result += vec
    result /= float(len(vecs))
    # TODO: Normalize?
    return result

