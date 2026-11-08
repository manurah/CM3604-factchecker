"""Member C code tests; controlled scores do not measure neural accuracy.

Run: python -m unittest discover -s member_c/tests -p test_evidence_processing.py -v
"""
from copy import deepcopy
from pathlib import Path
import unittest

from factcheck import WordRecallThresholdFactChecker
from factchecking_main import read_fact_examples, read_passages
from member_c.evidence_processing import (
    maximum_score, split_evidence_sentences, sentence_passages,
    PassageAggregatingWordFactChecker, SentenceAwareEntailmentFactChecker,
)

PROJECT = Path(__file__).resolve().parents[2]


class RecordingModel:
    def __init__(self, scores=None, default=0.0):
        self.scores = scores or {}
        self.default = default
        self.calls = []

    def check_entailment(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return self.scores.get(premise, self.default)


class PassageAggregationTests(unittest.TestCase):
    def test_no_scores_return_zero(self):
        self.assertEqual(maximum_score(iter(())), 0.0)

    def test_iterator_uses_maximum_not_mean_or_sum(self):
        self.assertEqual(maximum_score(iter((0.2, 0.8, 0.4))), 0.8)

    def test_supporting_passage_among_unrelated_passages(self):
        checker = PassageAggregatingWordFactChecker()
        passages = [{'text': 'volcanic lava'}, {'text': 'alpha beta gamma'}]
        self.assertEqual(checker.predict('alpha beta gamma delta', passages), 'S')
        self.assertEqual(checker.predict('alpha beta gamma delta', list(reversed(passages))), 'S')

    def test_separate_partial_passages_are_not_pooled(self):
        checker = PassageAggregatingWordFactChecker()
        self.assertEqual(checker.predict('alpha beta gamma delta', [
            {'text': 'alpha beta'}, {'text': 'gamma delta'}]), 'NS')

    def test_empty_evidence_and_empty_fact(self):
        checker = PassageAggregatingWordFactChecker()
        self.assertEqual(checker.predict('Ada wrote notes.', []), 'NS')
        self.assertEqual(checker.predict('', [{'text': 'Ada wrote notes.'}]), 'NS')

    def test_equal_and_below_threshold(self):
        checker = PassageAggregatingWordFactChecker(threshold=0.5)
        self.assertEqual(checker.predict('alpha beta gamma delta', [{'text': 'alpha beta'}]), 'S')
        self.assertEqual(checker.predict('alpha beta gamma delta', [{'text': 'alpha'}]), 'NS')

    def test_existing_scorers_and_preprocessing_options_preserved(self):
        for scorer in ('recall', 'jaccard', 'f1', 'rouge'):
            for use_nltk, remove_stopwords, use_stemming in ((True, True, True), (False, False, False)):
                with self.subTest(scorer=scorer, nltk=use_nltk):
                    options = dict(scorer=scorer, use_nltk=use_nltk,
                                   remove_stopwords=remove_stopwords, use_stemming=use_stemming)
                    original = WordRecallThresholdFactChecker(**options)
                    added = PassageAggregatingWordFactChecker(**options)
                    fact = 'Ada wrote useful notes.'
                    passages = [{'text': 'Ada writes useful notes.'}, {'text': 'Lava flows.'}]
                    self.assertEqual(added.preprocess(fact), original.preprocess(fact))
                    self.assertEqual(added.predict(fact, passages), original.predict(fact, passages))

    def test_default_predictions_match_existing_code_on_all_examples(self):
        examples = read_fact_examples(str(PROJECT / 'data/dev_labeled_ChatGPT.jsonl'),
                                     read_passages(str(PROJECT / 'data/passages_bm25_ChatGPT_humfacts.jsonl')))
        self.assertEqual(len(examples), 221)
        original = WordRecallThresholdFactChecker()
        added = PassageAggregatingWordFactChecker()
        for index, example in enumerate(examples):
            with self.subTest(example=index):
                self.assertEqual(added.predict(example.fact, example.passages),
                                 original.predict(example.fact, example.passages))


class SentenceProcessingTests(unittest.TestCase):
    def test_normal_prose(self):
        self.assertEqual(split_evidence_sentences('Ada wrote notes. She lived in London.'),
                         ['Ada wrote notes.', 'She lived in London.'])

    def test_adjacent_dataset_markers(self):
        self.assertEqual(split_evidence_sentences('<s>Ada wrote notes.</s><s>She lived in London.</s>'),
                         ['Ada wrote notes.', 'She lived in London.'])

    def test_marker_boundaries_without_punctuation(self):
        self.assertEqual(split_evidence_sentences('<s>First fragment</s><s>Second fragment</s>'),
                         ['First fragment', 'Second fragment'])

    def test_multiple_sentences_inside_one_block(self):
        self.assertEqual(split_evidence_sentences('<s>Ada wrote notes. She lived in London.</s>'),
                         ['Ada wrote notes.', 'She lived in London.'])

    def test_empty_text_and_marker_only_text(self):
        for text in ('', ' \n\t ', '<s> </s><s>\n</s>'):
            with self.subTest(text=text):
                self.assertEqual(split_evidence_sentences(text), [])

    def test_names_dates_and_negation_preserved(self):
        sentence = 'Roselyn Sánchez was not born on 18 July 1987.'
        self.assertEqual(split_evidence_sentences('<s>' + sentence + '</s>'), [sentence])

    def test_whitespace_cleaned_and_outside_text_retained(self):
        self.assertEqual(split_evidence_sentences('  Title\n<s>Ada  wrote\tnotes.</s>\nAn ending.'),
                         ['Title', 'Ada wrote notes.', 'An ending.'])

    def test_decimals_and_negation_not_split_or_removed(self):
        self.assertEqual(split_evidence_sentences('<s>Ada scored 3.14 points. She did not lose.</s>'),
                         ['Ada scored 3.14 points.', 'She did not lose.'])

    def test_passage_metadata_preserved_without_input_mutation(self):
        passages = [{'title': 'Ada', 'text': '<s>Ada wrote notes.</s><s>She lived in London.</s>'}]
        before = deepcopy(passages)
        iterator = sentence_passages(passages)
        self.assertIs(iter(iterator), iterator)
        self.assertEqual(list(iterator), [
            {'title': 'Ada', 'text': 'Ada wrote notes.'},
            {'title': 'Ada', 'text': 'She lived in London.'},
        ])
        self.assertEqual(passages, before)


class ExistingInferenceReuseTests(unittest.TestCase):
    def test_clean_premises_reach_existing_model_in_original_order(self):
        model = RecordingModel({'Ada wrote notes.': 0.8, 'She lived in London.': 0.2})
        checker = SentenceAwareEntailmentFactChecker(model, use_pruning=False)
        passages = [{'text': '<s>Ada wrote notes.</s><s>She lived in London.</s>'}]
        self.assertEqual(checker.predict('Ada wrote notes.', passages), 'S')
        self.assertEqual(model.calls, [('Ada wrote notes.', 'Ada wrote notes.'),
                                      ('She lived in London.', 'Ada wrote notes.')])
        self.assertEqual((checker.total_sentences, checker.pruned_sentences, checker.model_calls), (2, 0, 2))

    def test_original_pruning_still_filters_unrelated_evidence(self):
        model = RecordingModel(default=0.9)
        checker = SentenceAwareEntailmentFactChecker(model)
        self.assertEqual(checker.predict('Maya was born in London.', [
            {'text': '<s>Volcanoes release molten rock.</s>'}]), 'NS')
        self.assertEqual(model.calls, [])
        self.assertEqual((checker.total_sentences, checker.pruned_sentences, checker.model_calls), (1, 1, 0))

    def test_original_threshold_equality_is_preserved(self):
        checker = SentenceAwareEntailmentFactChecker(RecordingModel(default=0.38), use_pruning=False)
        self.assertEqual(checker.predict('A claim.', [{'text': '<s>Evidence.</s>'}]), 'S')

    def test_no_evidence_never_calls_model(self):
        model = RecordingModel(default=0.9)
        checker = SentenceAwareEntailmentFactChecker(model)
        self.assertEqual(checker.predict('Ada wrote notes.', [{'text': '<s> </s>'}]), 'NS')
        self.assertEqual(model.calls, [])

    def test_maximum_across_multiple_passages(self):
        model = RecordingModel({'First evidence.': 0.1, 'Second evidence.': 0.8, 'Third evidence.': 0.3})
        checker = SentenceAwareEntailmentFactChecker(model, use_pruning=False)
        self.assertEqual(checker.score_fact('A claim.', [
            {'text': '<s>First evidence.</s>'},
            {'text': '<s>Second evidence.</s><s>Third evidence.</s>'},
        ]), 0.8)
        self.assertEqual(checker.model_calls, 3)
