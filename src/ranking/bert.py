import numpy as np
from typing import List, Tuple

import spacy
from bert_embedding import BertEmbedding

nlp = spacy.load("de")
bert = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual')


def string_to_bert_embeddings(input: str) -> List[Tuple[str, np.array]]:
    doc = nlp(input)
    sentences = [sent.string.strip().lower() for sent in doc.sents]
    tokens_and_embeddings_per_sent = bert(sentences)
    result = []
    for tokens_and_embeddings in tokens_and_embeddings_per_sent:
        token_emb_tuple_list = zip(*tokens_and_embeddings)
        result += list(token_emb_tuple_list)
    return result


def score_cosine(qv: np.array, dv: np.array) -> float:
    cos_sim = np.dot(qv, dv) / (np.linalg.norm(qv) * np.linalg.norm(dv))  # val btwn. [1, -1]
    cos_sim += 1.  # val btwn. [2, 0]
    cos_sim /= 2.  # val btwn. [1, 0]
    return cos_sim


def word_vecs_to_document_vec(vecs: List[np.array], weights: np.array=None) -> np.array:
    assert(len(vecs) > 0)
    result = np.zeros(vecs[0].shape)
    for vec in vecs:
        # TODO: Consider weight
        result += vec
    result /= float(len(vecs))
    # TODO: Normalize?
    return result

