"""Member C's opt-in passage aggregation and evidence sentence processing.

Existing coursework files are unchanged. The classes below reuse the existing
lexical scorers and neural inference/pruning code. Run them explicitly with:

    python -m member_c.run_code --mode word_overlap
    python -m member_c.run_code --mode entailment

The supplied factchecking_main.py continues to instantiate its original classes.
"""
import re
from typing import Iterable, Iterator, List

import nltk

from factcheck import EntailmentFactChecker, WordRecallThresholdFactChecker


def maximum_score(scores: Iterable[float]) -> float:
    """Take the strongest nonnegative support score; no evidence gives zero."""
    best = 0.0
    for score in scores:
        best = max(best, score)
    return best


def split_evidence_sentences(text: str) -> List[str]:
    """Respect <s>/</s> boundaries, clean whitespace, then split normal prose.

    Marker boundaries matter even when joined without spaces or punctuation.
    Keep names, numbers, case and negation for entailment; do not apply lexical
    stemming or stopword removal to the model's evidence.
    """
    sentences = []
    for block in re.split(r"</?s>", text, flags=re.IGNORECASE):
        cleaned = " ".join(block.split())
        if cleaned:
            sentences.extend(sentence.strip()
                             for sentence in nltk.sent_tokenize(cleaned)
                             if sentence.strip())
    return sentences


def sentence_passages(passages: Iterable[dict]) -> Iterator[dict]:
    """Yield clean sentence passages without modifying the input dictionaries."""
    for passage in passages:
        for sentence in split_evidence_sentences(passage['text']):
            yield {**passage, 'text': sentence}


class PassageAggregatingWordFactChecker(WordRecallThresholdFactChecker):
    """Apply C's maximum aggregation using the existing lexical implementation.

    Inherit every scorer, preprocessing option and default classification
    threshold. No changes to Sandali's scorer or Dilshan's preprocessing.
    """
    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_words = self.preprocess(fact)
        scores = (
            self._compute_score(fact_words, self.preprocess(passage['text']),
                                fact, passage['text'])
            for passage in passages
        )
        return 'S' if maximum_score(scores) >= self.threshold else 'NS'


class SentenceAwareEntailmentFactChecker(EntailmentFactChecker):
    """Feed clean sentence evidence through the existing entailment checker.

    Only the evidence preparation is adapted. The inherited code still owns
    pruning, model calls, maximum entailment aggregation, counters and the S/NS
    threshold. Its normal NLTK split sees already separated sentences; no second
    neural inference is introduced. The generator avoids copying all passages.
    """
    def score_fact(self, fact: str, passages: List[dict]) -> float:
        return super().score_fact(fact, sentence_passages(passages))
