"""Check evaluation bookkeeping with a tiny synthetic backend, not DeBERTa."""
from types import SimpleNamespace
import unittest
import torch

from factcheck import FactExample
from member_c.evaluate_pruning import TracedEntailmentModel, metrics, run_variant


class EvaluationToolTests(unittest.TestCase):
    def test_metrics_include_ir_after_conversion(self):
        rows = [{'gold': 'S', 'prediction': 'S'},
                {'gold': 'NS', 'prediction': 'S'},
                {'gold': 'NS', 'prediction': 'NS'}]
        self.assertEqual(metrics(rows, 'prediction'), {
            'total': 3, 'correct': 2, 'accuracy': 2 / 3,
            'true_positives': 1, 'true_negatives': 1,
            'false_positives': 1, 'false_negatives': 0,
        })

    def test_empty_metrics_do_not_invent_accuracy(self):
        self.assertIsNone(metrics([], 'prediction')['accuracy'])

    def test_same_logits_drive_probability_and_argmax_comparisons(self):
        class TinyModel(torch.nn.Module):
            config = SimpleNamespace(label2id={'contradiction': 0, 'neutral': 1, 'entailment': 2})

            def forward(self, input_ids):
                self.assert_inference_state()
                logits = [-2.0, 0.0, 5.0] if input_ids.item() else [5.0, 0.0, -2.0]
                return SimpleNamespace(logits=torch.tensor([logits]))

            def assert_inference_state(self):
                if self.training or torch.is_grad_enabled():
                    raise AssertionError('Evaluation must disable training and gradients.')

        def tokenizer(premise, hypothesis, **kwargs):
            self.assertEqual(hypothesis, 'Paris is in France.')
            return {'input_ids': torch.tensor([[int(premise == hypothesis)]])}

        model = TinyModel()
        wrapper = TracedEntailmentModel(model, tokenizer)
        examples = [FactExample('Paris is in France.', [
            {'text': 'Paris is in France. Volcanoes release molten rock.'}], 'S'),
            FactExample('Paris is in France.', [], 'IR')]
        try:
            pruned = run_variant(examples, wrapper, True, 0.38, 0.05)
            unpruned = run_variant(examples, wrapper, False, 0.38, 0.05)
        finally:
            wrapper.close()
        self.assertEqual(pruned['model_calls'], 1)
        self.assertEqual(unpruned['model_calls'], 2)
        self.assertEqual(pruned['probability_metrics']['correct'], 2)
        self.assertEqual(unpruned['discrete_metrics']['correct'], 2)
        self.assertEqual(pruned['records'][0]['best_evidence']['premise'], 'Paris is in France.')
        self.assertEqual(pruned['records'][1]['gold'], 'NS')
        self.assertEqual(model._forward_hooks, {})
