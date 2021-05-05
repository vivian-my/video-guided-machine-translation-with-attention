from typing import List, Dict

import sacrebleu


def calculate_bleu(results: List[Dict[str, str]]) -> float:
    """
    Calculate the BLEU score for a collection of results.

    :param results: collection of dictionary with entries containing target sequences and generated sequences
    :return: BLEU score
    """

    candidates = [dct['decoded_sequence'] for dct in results]
    print(candidates)
    references = [[dct['target_sequence'] for dct in results]]

    bleu = sacrebleu.corpus_bleu(candidates, references)

    return bleu.score
