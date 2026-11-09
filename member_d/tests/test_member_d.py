"""Member D tests: real tensors with a synthetic backend, not DeBERTa accuracy.

Run: python -m unittest discover -s member_d/tests -v
"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from factcheck import EntailmentModel, FactExample
from member_d.bounded_entailment import BoundedEntailmentFactChecker, BoundedInference
from member_d.evaluation import binary_label, binary_metrics, select_errors, threshold_sweep
from member_d.run_code import neural_run


class TinyTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, premises, hypotheses, **options):
        if isinstance(premises, str):
            premises, hypotheses = [premises], [hypotheses]
        self.calls.append((list(premises), list(hypotheses), options))
        length = min(options['max_length'], max(len(p.split()) + len(h.split()) + 3
                                               for p, h in zip(premises, hypotheses)))
        ids = torch.zeros((len(premises), length), dtype=torch.long)
        for index, premise in enumerate(premises):
            ids[index, 0] = 1 if premise.startswith('Support') else 2 if premise.startswith('Maybe') else 0
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


class TinyModel(torch.nn.Module):
    config = SimpleNamespace(label2id={'contradiction': 0, 'neutral': 1, 'entailment': 2})

    def __init__(self, invalid=False):
        super().__init__()
        self.batch_sizes = []
        self.token_lengths = []
        self.invalid = invalid

    def forward(self, input_ids, attention_mask):
        if self.training or torch.is_grad_enabled():
            raise AssertionError('Inference must disable training and gradients.')
        self.batch_sizes.append(len(input_ids))
        self.token_lengths.append(input_ids.shape[1])
        values = [[3.0, 0.0, -2.0], [-2.0, 0.0, 4.0], [0.0, 1.0, 0.9]]
        logits = torch.tensor([values[index] for index in input_ids[:, 0].tolist()])
        if self.invalid:
            logits[:] = float('nan')
        return SimpleNamespace(logits=logits)


def wrapper():
    model, tokenizer = TinyModel(), TinyTokenizer()
    return EntailmentModel(model, tokenizer), model, tokenizer


class EvaluationTests(unittest.TestCase):
    def test_ir_mapping_and_denominator(self):
        result = binary_metrics(['S', 'NS', 'IR'], ['S', 'NS', 'S'])
        self.assertEqual((result['total'], result['correct'], result['false_positives']), (3, 2, 1))
        self.assertEqual(result['accuracy'], 2 / 3)

    def test_mismatched_lengths_are_not_silently_zipped(self):
        with self.assertRaises(ValueError):
            binary_metrics(['S', 'NS'], ['S'])

    def test_invalid_labels_and_predictions_rejected(self):
        with self.assertRaises(ValueError):
            binary_label('unknown')
        with self.assertRaises(ValueError):
            binary_metrics(['S'], ['IR'])

    def test_no_examples_do_not_imply_success(self):
        self.assertIsNone(binary_metrics([], [])['accuracy'])
        with self.assertRaises(ValueError):
            threshold_sweep([], [])

    def test_exact_threshold_equality(self):
        rows, _ = threshold_sweep(['S', 'NS'], [0.65, 0.649], [0.65])
        self.assertEqual(rows[0]['correct'], 2)

    def test_default_grid_includes_exact_hundredths_and_endpoints(self):
        rows, _ = threshold_sweep(['S'], [0.65])
        self.assertEqual(len(rows), 101)
        self.assertEqual((rows[0]['threshold'], rows[65]['threshold'], rows[-1]['threshold']), (0.0, 0.65, 1.0))

    def test_best_tie_prefers_declared_reference(self):
        _, best = threshold_sweep(['S', 'NS'], [0.9, 0.1], [0.4, 0.5, 0.65], reference=0.65)
        self.assertEqual(best['threshold'], 0.65)

    def test_symmetric_tie_is_not_decided_by_float_roundoff(self):
        _, best = threshold_sweep(['S', 'NS'], [0.9, 0.01], [0.1, 0.3], reference=0.2)
        self.assertEqual(best['threshold'], 0.1)

    def test_invalid_scores_thresholds_and_reference_rejected(self):
        for score in (float('nan'), float('inf'), -0.1, 1.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                threshold_sweep(['S'], [score])
        for cutoffs in ([], [float('nan')], [-0.1], [1.1]):
            with self.subTest(cutoffs=cutoffs), self.assertRaises(ValueError):
                threshold_sweep(['S'], [0.5], cutoffs)
        with self.assertRaises(ValueError):
            threshold_sweep(['S'], [0.5], reference=float('nan'))

    def test_error_selection_preserves_labels_and_does_not_invent_errors(self):
        records = [dict(example_id=3, original_label='S', gold='S', prediction='NS'),
                   dict(example_id=2, original_label='IR', gold='NS', prediction='S'),
                   dict(example_id=1, original_label='NS', gold='NS', prediction='NS')]
        before = deepcopy(records)
        errors = select_errors(records)
        self.assertEqual([(r['example_id'], r['error_type']) for r in errors], [(2, 'FP'), (3, 'FN')])
        self.assertEqual(records, before)
        self.assertEqual(errors[0]['original_label'], 'IR')

    def test_error_selection_limit_and_duplicate_ids(self):
        records = [dict(example_id=i, original_label='NS', prediction='S') for i in range(15)]
        self.assertEqual(len(select_errors(records)), 10)
        self.assertEqual(select_errors(records, per_type=0), [])
        with self.assertRaises(ValueError):
            select_errors([records[0], records[0]])

    def test_inconsistent_gold_is_rejected(self):
        with self.assertRaises(ValueError):
            select_errors([dict(example_id=0, original_label='IR', gold='S', prediction='S')])


class BoundedInferenceTests(unittest.TestCase):
    def test_batched_scores_match_original_scalar_math_and_class_mapping(self):
        base, _, _ = wrapper()
        pairs = [('Support evidence.', 'A claim.'), ('Contrary evidence.', 'A claim.'), ('Maybe evidence.', 'A claim.')]
        scalar = [base.check_entailment(*pair) for pair in pairs]
        batched, model, tokenizer = wrapper()
        inference = BoundedInference(batched, batch_size=2)
        results = list(inference.iter_scores(iter(pairs)))
        self.assertEqual(model.batch_sizes, [2, 1])
        for old, new in zip(scalar, results):
            self.assertAlmostEqual(old, new.probability, places=6)
            self.assertIsInstance(new.probability, float)
        self.assertEqual([r.argmax_entailment for r in results], [True, False, False])
        self.assertEqual(tokenizer.calls[0][:2], (['Support evidence.', 'Contrary evidence.'], ['A claim.', 'A claim.']))

    def test_input_is_consumed_one_bounded_batch_at_a_time(self):
        base, model, _ = wrapper()
        produced = []
        def pairs():
            for i in range(9):
                produced.append(i)
                yield 'Support evidence.', 'A claim.'
        inference = BoundedInference(base, batch_size=4)
        results = inference.iter_scores(pairs())
        first = next(results)
        self.assertEqual(len(produced), 4)
        self.assertIsInstance(first.probability, float)
        self.assertEqual(len(list(results)), 8)
        self.assertEqual(model.batch_sizes, [4, 4, 1])
        self.assertEqual((inference.max_pending_pairs, inference.forward_calls, inference.pair_count), (4, 3, 9))

    def test_long_pairs_use_the_512_token_limit(self):
        base, model, tokenizer = wrapper()
        list(BoundedInference(base).iter_scores([('Support ' * 2000, 'A claim.')]))
        self.assertEqual(model.token_lengths, [512])
        self.assertTrue(tokenizer.calls[0][2]['truncation'])
        self.assertTrue(tokenizer.calls[0][2]['padding'])

    def test_empty_input_does_not_call_model(self):
        base, model, _ = wrapper()
        self.assertEqual(list(BoundedInference(base).iter_scores([])), [])
        self.assertEqual(model.batch_sizes, [])

    def test_invalid_bounds_are_rejected(self):
        base, _, _ = wrapper()
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                BoundedInference(base, batch_size=size)
        for length in (0, 513, True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                BoundedInference(base, max_length=length)

    def test_nonfinite_model_outputs_do_not_become_fake_scores(self):
        base = EntailmentModel(TinyModel(invalid=True), TinyTokenizer())
        with self.assertRaises(ValueError):
            list(BoundedInference(base).iter_scores([('Support.', 'A claim.')]))


class AggregationTests(unittest.TestCase):
    def test_best_evidence_in_last_batch_is_not_missed(self):
        base, model, _ = wrapper()
        checker = BoundedEntailmentFactChecker(base, use_pruning=False, batch_size=2)
        prediction = checker.predict('A claim.', [{'text': '<s>Contrary one.</s><s>Contrary two.</s><s>Support last.</s>'}])
        self.assertEqual(prediction, 'S')
        self.assertEqual(checker.last_best_evidence, 'Support last.')
        self.assertEqual(model.batch_sizes, [2, 1])
        self.assertEqual((checker.total_sentences, checker.model_calls, checker.pruned_sentences), (3, 3, 0))

    def test_pruning_reuses_existing_overlap_rule(self):
        base, model, _ = wrapper()
        checker = BoundedEntailmentFactChecker(base)
        self.assertEqual(checker.predict('Maya was born in London.', [{'text': '<s>Volcanoes release molten rock.</s>'}]), 'NS')
        self.assertEqual(model.batch_sizes, [])
        self.assertEqual((checker.total_sentences, checker.pruned_sentences, checker.model_calls), (1, 1, 0))

    def test_best_evidence_and_discrete_state_reset_per_fact(self):
        base, _, _ = wrapper()
        checker = BoundedEntailmentFactChecker(base, use_pruning=False)
        self.assertEqual(checker.predict('A claim.', [{'text': '<s>Support.</s>'}]), 'S')
        self.assertEqual(checker.predict('A claim.', []), 'NS')
        self.assertIsNone(checker.last_best_evidence)
        self.assertFalse(checker.last_argmax_entailment)

    def test_probability_threshold_and_argmax_are_different_strategies(self):
        base, _, _ = wrapper()
        checker = BoundedEntailmentFactChecker(base, use_pruning=False)
        self.assertEqual(checker.predict('A claim.', [{'text': 'Maybe evidence.'}]), 'S')
        self.assertFalse(checker.last_argmax_entailment)

    def test_empty_marker_only_evidence_does_not_call_model(self):
        base, model, _ = wrapper()
        checker = BoundedEntailmentFactChecker(base)
        self.assertEqual(checker.predict('A claim.', [{'text': '<s> </s>'}]), 'NS')
        self.assertEqual(model.batch_sizes, [])

    def test_invalid_thresholds_rejected(self):
        base, _, _ = wrapper()
        with self.assertRaises(ValueError):
            BoundedEntailmentFactChecker(base, threshold=0)
        with self.assertRaises(ValueError):
            BoundedEntailmentFactChecker(base, pruning_threshold=float('nan'))

    def test_runner_writes_reconciled_metrics_and_actual_errors(self):
        base, _, _ = wrapper()
        examples = [FactExample('A claim.', [{'text': '<s>Contrary.</s><s>Support.</s>'}], 'S'),
                    FactExample('A claim.', [{'text': 'Support.'}], 'IR'),
                    FactExample('A claim.', [], 'S')]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            result = neural_run(examples, base, output, False, 0.38, 2)
            stored = json.loads((output / 'unpruned.json').read_text())
            errors = json.loads((output / 'unpruned_selected_errors.json').read_text())
            self.assertEqual(result['probability_metrics']['total'], 3)
            self.assertEqual(result['probability_metrics']['correct'], 1)
            self.assertEqual((result['sentence_pairs'], result['model_forward_calls']), (3, 2))
            self.assertEqual(stored['records'][1]['gold'], 'NS')
            self.assertIsNone(stored['records'][2]['best_evidence'])
            self.assertEqual([(r['example_id'], r['error_type']) for r in errors['records']], [(1, 'FP'), (2, 'FN')])
            self.assertTrue((output / 'unpruned_thresholds.csv').is_file())
