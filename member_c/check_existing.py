"""Audit existing coursework code; does not implement or modify assessed work.

Run with the repository virtual environment and NLTK_DATA configured.
Synthetic model scores exercise control flow only, not DeBERTa accuracy.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path
import sys
import unittest
from collections import Counter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import nltk
from factcheck import EntailmentFactChecker, WordRecallThresholdFactChecker
from factchecking_main import read_fact_examples, read_passages


class RecordingModel:
    def __init__(self, scores=None, default=0.0):
        self.scores = scores or {}
        self.default = default
        self.calls = []

    def check_entailment(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return self.scores.get(premise, self.default)


class ExistingBehaviour(unittest.TestCase):
    def test_recall_arithmetic(self):
        checker = WordRecallThresholdFactChecker(scorer='recall')
        self.assertEqual(checker._compute_score({'a', 'b'}, {'b', 'c', 'd'}, '', ''), 0.5)

    def test_jaccard_arithmetic(self):
        checker = WordRecallThresholdFactChecker(scorer='jaccard')
        self.assertEqual(checker._compute_score({'a', 'b'}, {'b', 'c', 'd'}, '', ''), 0.25)

    def test_f1_arithmetic(self):
        checker = WordRecallThresholdFactChecker(scorer='f1')
        self.assertAlmostEqual(checker._compute_score({'a', 'b'}, {'b', 'c', 'd'}, '', ''), 0.4)

    def test_rouge_arithmetic(self):
        checker = WordRecallThresholdFactChecker(scorer='rouge')
        self.assertAlmostEqual(checker._compute_score({'red', 'blue'}, {'red', 'green'}, 'red blue', 'red green'), 0.5)

    def test_empty_fact_scores_zero(self):
        for scorer in ['recall', 'jaccard', 'f1', 'rouge']:
            with self.subTest(scorer=scorer):
                checker = WordRecallThresholdFactChecker(scorer=scorer)
                self.assertEqual(checker._compute_score(set(), {'a'}, '', 'a'), 0.0)

    def test_unknown_scorer_rejected(self):
        checker = WordRecallThresholdFactChecker(scorer='invalid')
        with self.assertRaises(ValueError):
            checker._compute_score({'a'}, {'a'}, 'a', 'a')

    def test_lexical_empty_passages(self):
        self.assertEqual(WordRecallThresholdFactChecker().predict('Ada wrote notes.', []), 'NS')

    def test_lexical_single_support_among_unrelated_passages(self):
        passages = [{'text': 'volcanic lava'}, {'text': 'alpha beta gamma'}, {'text': 'green grass'}]
        checker = WordRecallThresholdFactChecker()
        self.assertEqual(checker.predict('alpha beta gamma delta', passages), 'S')
        self.assertEqual(checker.predict('alpha beta gamma delta', list(reversed(passages))), 'S')

    def test_lexical_partial_passages_not_added(self):
        checker = WordRecallThresholdFactChecker()
        self.assertEqual(checker.predict('alpha beta gamma delta', [{'text': 'alpha beta'}, {'text': 'gamma delta'}]), 'NS')

    def test_lexical_threshold_equal_and_below(self):
        checker = WordRecallThresholdFactChecker(threshold=0.5)
        self.assertEqual(checker.predict('alpha beta gamma delta', [{'text': 'alpha beta'}]), 'S')
        self.assertEqual(checker.predict('alpha beta gamma delta', [{'text': 'alpha'}]), 'NS')

    def test_entailment_max_not_average_or_sum(self):
        model = RecordingModel({'First sentence.': 0.2, 'Second sentence.': 0.6, 'Third sentence.': 0.4})
        checker = EntailmentFactChecker(model, use_pruning=False)
        score = checker.score_fact('The claim.', [{'text': 'First sentence. Second sentence.'}, {'text': 'Third sentence.'}])
        self.assertEqual(score, 0.6)
        self.assertEqual(checker.model_calls, 3)
        self.assertTrue(all(hypothesis == 'The claim.' for _, hypothesis in model.calls))

    def test_entailment_threshold_equality(self):
        checker = EntailmentFactChecker(RecordingModel(default=0.38), use_pruning=False)
        self.assertEqual(checker.predict('The claim.', [{'text': 'Evidence.'}]), 'S')

    def test_entailment_below_threshold(self):
        checker = EntailmentFactChecker(RecordingModel(default=0.379), use_pruning=False)
        self.assertEqual(checker.predict('The claim.', [{'text': 'Evidence.'}]), 'NS')

    def test_entailment_no_evidence(self):
        model = RecordingModel(default=0.9)
        checker = EntailmentFactChecker(model)
        self.assertEqual(checker.predict('Ada wrote notes.', []), 'NS')
        self.assertEqual(model.calls, [])

    def test_entailment_whitespace_only(self):
        model = RecordingModel(default=0.9)
        checker = EntailmentFactChecker(model)
        self.assertEqual(checker.predict('Ada wrote notes.', [{'text': ' \n\t '}]), 'NS')
        self.assertEqual(model.calls, [])

    def test_zero_overlap_pruned_without_inference(self):
        model = RecordingModel(default=0.9)
        checker = EntailmentFactChecker(model)
        self.assertEqual(checker.predict('Maya was born in London.', [{'text': 'Volcanoes release molten rock.'}]), 'NS')
        self.assertEqual(model.calls, [])
        self.assertEqual((checker.total_sentences, checker.pruned_sentences, checker.model_calls), (1, 1, 0))

    def test_pruning_disabled_reaches_model(self):
        model = RecordingModel(default=0.9)
        checker = EntailmentFactChecker(model, use_pruning=False)
        self.assertEqual(checker.predict('Maya was born in London.', [{'text': 'Volcanoes release molten rock.'}]), 'S')
        self.assertEqual(len(model.calls), 1)

    def test_natural_evidence_case_preserved(self):
        model = RecordingModel(default=0.9)
        checker = EntailmentFactChecker(model)
        checker.predict('Paris is in France.', [{'text': 'Paris is in France.'}])
        self.assertEqual(model.calls, [('Paris is in France.', 'Paris is in France.')])

    def test_sentence_boundaries_without_markup(self):
        model = RecordingModel()
        checker = EntailmentFactChecker(model, use_pruning=False)
        checker.score_fact('Ada wrote notes.', [{'text': 'Ada wrote notes. She lived in London.'}])
        self.assertEqual([p for p, _ in model.calls], ['Ada wrote notes.', 'She lived in London.'])

    def test_sentence_boundaries_and_cleaning_with_dataset_markup(self):
        model = RecordingModel()
        checker = EntailmentFactChecker(model, use_pruning=False)
        checker.score_fact('Ada wrote notes.', [{'text': '<s>Ada wrote notes.</s><s>She lived in London.</s>'}])
        self.assertEqual([p for p, _ in model.calls], ['Ada wrote notes.', 'She lived in London.'])


def inspect_data_and_cache():
    messages = io.StringIO()
    with contextlib.redirect_stdout(messages):
        passages = read_passages(str(PROJECT / 'data/passages_bm25_ChatGPT_humfacts.jsonl'))
        examples = read_fact_examples(str(PROJECT / 'data/dev_labeled_ChatGPT.jsonl'), passages)
    labelled = [json.loads(line) for line in (PROJECT / 'data/dev_labeled_ChatGPT.jsonl').open()]
    retrieved = [json.loads(line) for line in (PROJECT / 'data/passages_bm25_ChatGPT_humfacts.jsonl').open()]
    model = RecordingModel()
    checker = EntailmentFactChecker(model)
    sentence_count = marked_sentences = marked_passages = total_passages = 0
    for example in examples:
        for passage in example.passages:
            total_passages += 1
            text = ' '.join(passage['text'].split())
            marked_passages += '<s>' in text or '</s>' in text
            for sentence in nltk.sent_tokenize(text):
                sentence_count += 1
                marked_sentences += '<s>' in sentence or '</s>' in sentence
        checker.score_fact(example.fact, example.passages)
    cache = json.loads((PROJECT / 'results/entailment_pruning_005_scores.json').read_text())
    fingerprints = {}
    for key, relative in [('factcheck_sha256', 'factcheck.py'), ('driver_sha256', 'factchecking_main.py'), ('labels_sha256', 'data/dev_labeled_ChatGPT.jsonl'), ('passages_sha256', 'data/passages_bm25_ChatGPT_humfacts.jsonl')]:
        raw = (PROJECT / relative).read_bytes()
        lf = raw.replace(b'\r\n', b'\n')
        variants = {'current_bytes': raw, 'windows_CRLF': lf.replace(b'\n', b'\r\n')}
        if relative == 'factcheck.py':
            old = lf.replace(b'threshold: float = 0.38', b'threshold: float = 0.5')
            variants['windows_CRLF_with_entailment_default_0.5'] = old.replace(b'\n', b'\r\n')
        fingerprints[relative] = {
            'current_sha256': hashlib.sha256(raw).hexdigest(),
            'saved_sha256': cache['signature'][key],
            'matching_variants': [name for name, data in variants.items() if hashlib.sha256(data).hexdigest() == cache['signature'][key]],
        }
    scores = cache['records']
    csv_rows = list(csv.DictReader((PROJECT / 'results/entailment_pruning_005_thresholds.csv').open()))
    csv_checks = []
    for row in csv_rows:
        threshold = float(row['threshold'])
        c = Counter(('true_positives' if r['gold'] == 'S' else 'false_positives') if r['score'] >= threshold else ('false_negatives' if r['gold'] == 'S' else 'true_negatives') for r in scores)
        csv_checks.append(all(int(row[k]) == c[k] for k in ('true_positives', 'true_negatives', 'false_positives', 'false_negatives')) and int(row['correct']) == c['true_positives'] + c['true_negatives'] and int(row['total']) == len(scores) and math.isclose(float(row['accuracy']), (c['true_positives'] + c['true_negatives']) / len(scores)))
    metrics = {}
    for threshold in (0.38, 0.5):
        c = Counter(('TP' if r['gold'] == 'S' else 'FP') if r['score'] >= threshold else ('FN' if r['gold'] == 'S' else 'TN') for r in scores)
        metrics[str(threshold)] = dict(c, correct=c['TP'] + c['TN'], total=len(scores), accuracy=(c['TP'] + c['TN']) / len(scores))
    return {
        'evidence_kind': 'Existing-code audit; cached score arithmetic is not fresh model inference.',
        'git_head': subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip(),
        'data': {'biographies': len(labelled), 'retrieved_rows': len(retrieved), 'unique_retrieved_facts': len(passages), 'loaded_examples': len(examples), 'labels': dict(Counter(e.label for e in examples)), 'driver_binary_labels': dict(Counter('NS' if e.label == 'IR' else e.label for e in examples)), 'loader_messages': messages.getvalue(), 'passages_per_example': dict(Counter(len(e.passages) for e in examples))},
        'segmentation': {'evaluated_passages': total_passages, 'passages_containing_s_markup': marked_passages, 'current_sentence_count': sentence_count, 'sentences_containing_s_markup': marked_sentences, 'model_inputs_containing_s_markup_after_pruning': sum('<s>' in p or '</s>' in p for p, _ in model.calls), 'first_retained_model_input': model.calls[0][0] if model.calls else None},
        'pruning_structure_only': {'total_sentences': checker.total_sentences, 'pruned_sentences': checker.pruned_sentences, 'retained_model_calls': checker.model_calls, 'fraction_pruned': checker.pruned_sentences / checker.total_sentences, 'real_model_used': False, 'accuracy_or_model_runtime_comparison_available': False},
        'cache': {'signature': cache['signature'], 'fingerprints': fingerprints, 'records': len(scores), 'records_match_current_ordered_data': len(examples) == len(scores) and all(r['example_id'] == i and r['fact'] == e.fact and r['original_label'] == e.label and r['gold'] == ('NS' if e.label == 'IR' else e.label) and r['passages'] == e.passages for i, (r, e) in enumerate(zip(scores, examples))), 'scores_finite_and_in_range': all(math.isfinite(r['score']) and 0 <= r['score'] <= 1 for r in scores), 'csv_rows_checked': len(csv_rows), 'all_csv_rows_match_scores': all(csv_checks), 'metrics_from_saved_scores_only': metrics, 'reported_old_run_metadata_not_reverified': {k: cache[k] for k in ('evaluation_seconds', 'total_sentences', 'pruned_sentences', 'model_calls')}},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for audit evidence; existing directories are refused.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; historical evidence is preserved.')
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True)
    summary = inspect_data_and_cache()
    with (OUTPUT / 'behaviour-checks.txt').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExistingBehaviour))
    summary['behaviour_checks'] = {'executed': result.testsRun, 'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped), 'failed': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'failures': [{'test': test.id(), 'traceback': error} for test, error in result.failures + result.errors]}
    (OUTPUT / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
