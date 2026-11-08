"""Reproduce C's pruning comparison without changing the coursework driver.

Both variants use the same checkpoint and decision threshold. A forward hook
also records the discrete decision from the SAME logits, without a second model
call. Output directories must be new so historical evidence is never overwritten.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from factcheck import EntailmentFactChecker, EntailmentModel
from factchecking_main import read_fact_examples, read_passages

MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def metrics(records, prediction_key):
    tp = tn = fp = fn = 0
    for record in records:
        gold, prediction = record['gold'], record[prediction_key]
        if gold == 'S':
            if prediction == 'S':
                tp += 1
            else:
                fn += 1
        elif prediction == 'S':
            fp += 1
        else:
            tn += 1
    total = len(records)
    return dict(total=total, correct=tp + tn,
                accuracy=(tp + tn) / total if total else None,
                true_positives=tp, true_negatives=tn,
                false_positives=fp, false_negatives=fn)


class TracedEntailmentModel(EntailmentModel):
    """Reuse submitted inference; retain only scalar decisions and text."""
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, cuda=False)
        self.trace = []
        self.last_class = None
        self.hook = self.model.register_forward_hook(self._observe_logits)

    def _observe_logits(self, module, inputs, outputs):
        self.last_class = int(outputs.logits.argmax(dim=-1).item())

    def check_entailment(self, premise, hypothesis):
        self.last_class = None
        score = super().check_entailment(premise, hypothesis)
        if self.last_class is None:
            raise RuntimeError('Inference hook did not observe the model output.')
        self.trace.append({'premise': premise, 'score': score,
                           'argmax_is_entailment': self.last_class == self.entailment_index})
        return score

    def close(self):
        self.hook.remove()


def run_variant(examples, wrapper, use_pruning, threshold, pruning_threshold):
    checker = EntailmentFactChecker(wrapper, threshold=threshold,
                                   pruning_threshold=pruning_threshold,
                                   use_pruning=use_pruning)
    records = []
    started = perf_counter()
    for index, example in enumerate(examples):
        wrapper.trace = []
        score = checker.score_fact(example.fact, example.passages)
        best = max(wrapper.trace, key=lambda pair: pair['score'], default=None)
        records.append({
            'example_id': index, 'fact': example.fact,
            'original_label': example.label,
            'gold': 'NS' if example.label == 'IR' else example.label,
            'score': score, 'prediction': 'S' if score >= threshold else 'NS',
            'discrete_prediction': 'S' if any(p['argmax_is_entailment'] for p in wrapper.trace) else 'NS',
            'best_evidence': best, 'examined_pairs': len(wrapper.trace),
            'passages': example.passages,
        })
        if (index + 1) % 25 == 0:
            print(f'{"pruned" if use_pruning else "unpruned"}: {index + 1}/{len(examples)} claims', flush=True)
    elapsed = perf_counter() - started
    assert checker.total_sentences == checker.pruned_sentences + checker.model_calls
    assert checker.model_calls == sum(r['examined_pairs'] for r in records)
    return {
        'use_pruning': use_pruning, 'threshold': threshold,
        'pruning_threshold': pruning_threshold,
        'scoring_seconds_excluding_model_load_and_file_writes': elapsed,
        'total_sentences': checker.total_sentences,
        'pruned_sentences': checker.pruned_sentences,
        'model_calls': checker.model_calls,
        'probability_metrics': metrics(records, 'prediction'),
        'discrete_metrics': metrics(records, 'discrete_prediction'),
        'records': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing directories are refused.')
    parser.add_argument('--model', default=MODEL, help='Public checkpoint name or a local checkpoint directory.')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--limit', type=int, help='Debug subset only; omit for full evaluation.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.38)
    parser.add_argument('--pruning-threshold', type=float, default=0.05)
    parser.add_argument('--order', choices=['pruned-first', 'unpruned-first'], default='pruned-first')
    args = parser.parse_args()
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('threads and limit must be positive')
    if not 0 < args.threshold <= 1 or not 0 <= args.pruning_threshold <= 1:
        parser.error('threshold must be in (0, 1]; pruning threshold must be in [0, 1]')
    if args.output.exists():
        parser.error('Output already exists. Choose a new directory to preserve evidence.')
    args.output.mkdir(parents=True)
    manifest = {
        'status': 'started', 'requested_model': args.model,
        'device': 'cpu', 'threads': args.threads,
        'threshold': args.threshold, 'pruning_threshold': args.pruning_threshold,
        'order': args.order, 'variants_completed': [], 'full_dataset_run': False,
        'limitations': 'Single ordered timings include tokenization, Python checks and gc.collect; no warm-up is excluded. Repeat in reverse order before attributing small timing differences to pruning. Discrete decisions use the same retained evidence as each probability variant.',
        'file_sha256': {str(relative): hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest() for relative in (
            Path('factcheck.py'), Path('factchecking_main.py'), Path('data/dev_labeled_ChatGPT.jsonl'),
            Path('data/passages_bm25_ChatGPT_humfacts.jsonl'), Path('member_c/evaluate_pruning.py'))},
        'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'nltk', 'numpy')},
    }
    write_json(args.output / 'manifest.json', manifest)
    wrapper = None
    try:
        torch.set_num_threads(args.threads)
        examples = read_fact_examples(str(PROJECT / 'data/dev_labeled_ChatGPT.jsonl'),
                                     read_passages(str(PROJECT / 'data/passages_bm25_ChatGPT_humfacts.jsonl')))
        full_count = len(examples)
        if args.limit:
            examples = examples[:args.limit]
        if not examples:
            raise ValueError('No examples were loaded.')
        manifest.update(full_dataset_count=full_count, evaluated_count=len(examples),
                        full_dataset_requested=len(examples) == full_count)
        started = perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=args.local_files_only)
        manifest.update(model_load_seconds=perf_counter() - started,
                        checkpoint_commit=getattr(model.config, '_commit_hash', None),
                        label2id=model.config.label2id)
        wrapper = TracedEntailmentModel(model, tokenizer)
        variants = [True, False] if args.order == 'pruned-first' else [False, True]
        for use_pruning in variants:
            name = 'pruned' if use_pruning else 'unpruned'
            result = run_variant(examples, wrapper, use_pruning, args.threshold, args.pruning_threshold)
            write_json(args.output / f'{name}.json', result)
            manifest['variants_completed'].append(name)
            write_json(args.output / 'manifest.json', manifest)
            print(name, result['probability_metrics'], flush=True)
        manifest['status'] = 'completed'
        manifest['full_dataset_run'] = len(examples) == full_count
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if wrapper is not None:
            wrapper.close()
        write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
