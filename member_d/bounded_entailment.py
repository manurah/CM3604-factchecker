"""Opt-in Member D batching and maximum-score evidence aggregation.

Existing files and their inference implementation remain unchanged. This
adapter uses the SAME loaded model/tokenizer, premise-first ordering, softmax
and configured entailment index, but processes at most batch_size pairs in a
forward pass. Member C supplies cleaned sentences; the existing overlap scorer
and low-threshold pruning rule are reused.
"""
from itertools import islice
import math
from typing import NamedTuple

import torch

from factcheck import EntailmentFactChecker
from member_c.evidence_processing import sentence_passages


class PairScore(NamedTuple):
    premise: str
    probability: float
    argmax_entailment: bool


class BoundedInference:
    def __init__(self, wrapper, batch_size=4, max_length=512):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        if not isinstance(max_length, int) or isinstance(max_length, bool) or not 1 <= max_length <= 512:
            raise ValueError('max_length must be an integer between 1 and 512.')
        self.wrapper = wrapper
        self.batch_size = batch_size
        self.max_length = max_length
        self.forward_calls = 0
        self.pair_count = 0
        self.max_pending_pairs = 0
        self.wrapper.model.eval()

    def iter_scores(self, pairs):
        iterator = iter(pairs)
        while True:
            batch = list(islice(iterator, self.batch_size))
            if not batch:
                return
            self.max_pending_pairs = max(self.max_pending_pairs, len(batch))
            premises, hypotheses = zip(*batch)
            with torch.no_grad():
                inputs = self.wrapper.tokenizer(list(premises), list(hypotheses),
                                                return_tensors='pt', truncation=True,
                                                max_length=self.max_length, padding=True)
                if self.wrapper.cuda:
                    inputs = {key: value.to('cuda') for key, value in inputs.items()}
                outputs = self.wrapper.model(**inputs)
                logits = outputs.logits
                if logits.ndim != 2 or logits.shape[0] != len(batch):
                    raise ValueError('Model output is not one class-logit vector per pair.')
                probabilities = torch.softmax(logits, dim=-1)
                scores = probabilities[:, self.wrapper.entailment_index].cpu().tolist()
                winning_classes = logits.argmax(dim=-1).cpu().tolist()
            # Release input/output tensors before yielding Python scalar results.
            del inputs, outputs, logits, probabilities
            self.forward_calls += 1
            self.pair_count += len(batch)
            for premise, score, winning_class in zip(premises, scores, winning_classes):
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Model returned a non-finite or invalid probability.')
                yield PairScore(premise, score, winning_class == self.wrapper.entailment_index)


class BoundedEntailmentFactChecker(EntailmentFactChecker):
    """Stream retained sentences into bounded inference; inspect every batch.

    predict(), the decision threshold and the overlap checker are inherited.
    model_calls counts sentence pairs, while inference.forward_calls counts
    actual minibatch forwards. These are intentionally different quantities.
    """
    def __init__(self, ent_model, threshold=0.38, pruning_threshold=0.05,
                 use_pruning=True, batch_size=4, max_length=512):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('Entailment threshold must be in (0, 1].')
        if not math.isfinite(pruning_threshold) or not 0 <= pruning_threshold <= 1:
            raise ValueError('Pruning threshold must be in [0, 1].')
        super().__init__(ent_model, threshold, pruning_threshold, use_pruning)
        self.inference = BoundedInference(ent_model, batch_size, max_length)
        self.last_best_evidence = None
        self.last_argmax_entailment = False

    def _retained_pairs(self, fact, passages):
        fact_words = self.overlap_checker.preprocess(fact) if self.use_pruning else None
        for passage in sentence_passages(passages):
            sentence = passage['text']
            self.total_sentences += 1
            if self.use_pruning and fact_words:
                words = self.overlap_checker.preprocess(sentence)
                overlap = self.overlap_checker._compute_score(fact_words, words, fact, sentence)
                if overlap < self.pruning_threshold:
                    self.pruned_sentences += 1
                    continue
            yield sentence, fact

    def score_fact(self, fact, passages):
        best = 0.0
        self.last_best_evidence = None
        self.last_argmax_entailment = False
        for result in self.inference.iter_scores(self._retained_pairs(fact, passages)):
            self.model_calls += 1
            self.last_argmax_entailment |= result.argmax_entailment
            if self.last_best_evidence is None or result.probability > best:
                best = result.probability
                self.last_best_evidence = result.premise
        return best
