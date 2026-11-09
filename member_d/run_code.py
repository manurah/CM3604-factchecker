"""Member D experiments using opt-in components; original files stay unchanged.

python -m member_d.run_code --mode word_overlap --ablation --output /tmp/lexical-run
python -m member_d.run_code --mode entailment --compare-pruning --output /tmp/nli-run

Every output directory must be new. Subset runs and failed checkpoint loads
cannot be reported as completed full evaluation. Report/viva writing is separate.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factcheck import EntailmentModel
from factchecking_main import read_fact_examples, read_passages
from member_c.evidence_processing import PassageAggregatingWordFactChecker, maximum_score
from member_d.bounded_entailment import BoundedEntailmentFactChecker
from member_d.evaluation import binary_label, binary_metrics, select_errors, threshold_sweep

PROJECT = Path(__file__).resolve().parents[1]
MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def write_thresholds(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lexical_run(examples, output, threshold, ablation):
    configurations = [(scorer, True, True, True) for scorer in ('recall', 'jaccard', 'f1', 'rouge')]
    if ablation:
        configurations.extend(('recall', *flags) for flags in product((False, True), repeat=3)
                              if flags != (True, True, True))
    summary = []
    labels = [example.label for example in examples]
    for scorer, nltk, stopwords, stemming in configurations:
        checker = PassageAggregatingWordFactChecker(
            threshold=threshold, scorer=scorer, use_nltk=nltk,
            remove_stopwords=stopwords, use_stemming=stemming)
        # Only preprocessing is cached; no gold-label lookup or outcome cache.
        preprocess = lru_cache(maxsize=4096)(checker.preprocess)
        records = []
        started = perf_counter()
        for index, example in enumerate(examples):
            fact_words = preprocess(example.fact)
            score = maximum_score(checker._compute_score(
                fact_words, preprocess(passage['text']), example.fact, passage['text'])
                for passage in example.passages)
            records.append(dict(example_id=index, fact=example.fact,
                                original_label=example.label, gold=binary_label(example.label),
                                score=score, prediction='S' if score >= threshold else 'NS'))
        scoring_seconds = perf_counter() - started
        started = perf_counter()
        rows, best = threshold_sweep(labels, [r['score'] for r in records], reference=threshold)
        sweep_seconds = perf_counter() - started
        name = f'{scorer}_nltk{int(nltk)}_stop{int(stopwords)}_stem{int(stemming)}'
        result = dict(configuration=name, scorer=scorer, use_nltk=nltk,
                      remove_stopwords=stopwords, use_stemming=stemming,
                      threshold=threshold, scoring_seconds=scoring_seconds,
                      sweep_seconds=sweep_seconds, best_threshold=best,
                      fixed_threshold_metrics=binary_metrics(labels, [r['prediction'] for r in records]))
        write_json(output / f'{name}.json', dict(result, records=records))
        write_thresholds(output / f'{name}_thresholds.csv', rows)
        summary.append(result)
        print(name, result['fixed_threshold_metrics'], 'best cutoff:', best['threshold'], flush=True)
    return summary


def neural_run(examples, wrapper, output, use_pruning, threshold, batch_size):
    checker = BoundedEntailmentFactChecker(wrapper, threshold=threshold,
                                         use_pruning=use_pruning, batch_size=batch_size)
    records = []
    started = perf_counter()
    for index, example in enumerate(examples):
        score = checker.score_fact(example.fact, example.passages)
        records.append(dict(example_id=index, fact=example.fact,
                            original_label=example.label, gold=binary_label(example.label),
                            score=score, prediction='S' if score >= threshold else 'NS',
                            discrete_prediction='S' if checker.last_argmax_entailment else 'NS',
                            best_evidence=checker.last_best_evidence, passages=example.passages))
        if (index + 1) % 25 == 0:
            print(f'{"pruned" if use_pruning else "unpruned"}: {index + 1}/{len(examples)} claims', flush=True)
    scoring_seconds = perf_counter() - started
    if checker.total_sentences != checker.pruned_sentences + checker.model_calls:
        raise RuntimeError('Sentence filtering counts do not reconcile.')
    if checker.model_calls != checker.inference.pair_count:
        raise RuntimeError('Pair counts do not reconcile with minibatch inference.')
    labels = [e.label for e in examples]
    rows, best = threshold_sweep(labels, [r['score'] for r in records],
                                 thresholds=[step / 100 for step in range(1, 101)], reference=threshold)
    name = 'pruned' if use_pruning else 'unpruned'
    result = dict(use_pruning=use_pruning, pruning_threshold=checker.pruning_threshold,
                  threshold=threshold, batch_size=batch_size, max_length=512,
                  scoring_seconds=scoring_seconds, total_sentences=checker.total_sentences,
                  pruned_sentences=checker.pruned_sentences, sentence_pairs=checker.model_calls,
                  model_forward_calls=checker.inference.forward_calls,
                  max_pending_pairs=checker.inference.max_pending_pairs,
                  probability_metrics=binary_metrics(labels, [r['prediction'] for r in records]),
                  discrete_metrics=binary_metrics(labels, [r['discrete_prediction'] for r in records]),
                  best_threshold=best)
    write_json(output / f'{name}.json', dict(result, records=records))
    write_json(output / f'{name}_selected_errors.json', {
        'source': f'{name}.json', 'threshold': threshold,
        'selection': 'First up to 10 FP and 10 FN in driver-ID order; no invented errors.',
        'records': select_errors(records),
    })
    write_thresholds(output / f'{name}_thresholds.csv', rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['word_overlap', 'entailment'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ablation', action='store_true', help='Also evaluate all 8 lexical preprocessing combinations.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--compare-pruning', action='store_true')
    parser.add_argument('--no-pruning', action='store_true')
    parser.add_argument('--reverse-order', action='store_true', help='Unpruned comparison first; use a new output directory.')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--limit', type=int, help='Debug subset; omit for full evaluation.')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('batch-size, threads and limit must be positive.')
    threshold = args.threshold if args.threshold is not None else (0.65 if args.mode == 'word_overlap' else 0.38)
    if not 0 < threshold <= 1:
        parser.error('threshold must be in (0, 1].')
    if args.compare_pruning and args.no_pruning:
        parser.error('Choose comparison or no-pruning, not both.')
    if args.mode == 'word_overlap' and (args.cuda or args.compare_pruning or args.no_pruning or args.reverse_order):
        parser.error('CUDA and pruning options apply only to entailment mode.')
    if args.mode == 'entailment' and args.ablation:
        parser.error('Lexical ablation applies only to word_overlap mode.')
    if args.output.exists():
        parser.error('Output directory exists. Choose a new one to preserve evidence.')
    args.output.mkdir(parents=True)
    paths = ('factcheck.py', 'factchecking_main.py', 'member_c/evidence_processing.py',
             'member_d/evaluation.py', 'member_d/bounded_entailment.py', 'member_d/run_code.py',
             'data/dev_labeled_ChatGPT.jsonl', 'data/passages_bm25_ChatGPT_humfacts.jsonl')
    manifest = dict(status='started', mode=args.mode, full_dataset_run=False,
                    completed_runs=[], threshold=threshold,
                    git_head=subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'],
                                            check=True, capture_output=True, text=True).stdout.strip(),
                    source_sha256={p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest() for p in paths},
                    python_version=sys.version.split()[0],
                    package_versions={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'nltk', 'numpy')},
                    device='cuda' if args.cuda else 'cpu', threads=args.threads,
                    requested_model=args.model if args.mode == 'entailment' else None,
                    batch_size=args.batch_size if args.mode == 'entailment' else None,
                    limitations='Development-set tuning is not independent test-set evaluation. Neural scoring timings include preprocessing, record assembly and progress logging but exclude checkpoint loading, threshold sweeps and file writes; reverse variant order before attributing small timing differences to pruning. Minibatch forwards and sentence-pair counts differ.')
    write_json(args.output / 'manifest.json', manifest)
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
        if args.mode == 'word_overlap':
            manifest['lexical_summary'] = lexical_run(examples, args.output, threshold, args.ablation)
            manifest['completed_runs'] = [r['configuration'] for r in manifest['lexical_summary']]
        else:
            started = perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
            model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=args.local_files_only)
            if args.cuda:
                model.to('cuda')
            wrapper = EntailmentModel(model, tokenizer, args.cuda)
            manifest.update(model_load_seconds=perf_counter() - started, requested_model=args.model,
                            checkpoint_commit=getattr(model.config, '_commit_hash', None),
                            label2id=model.config.label2id, batch_size=args.batch_size)
            variants = [True, False] if args.compare_pruning else [not args.no_pruning]
            if args.reverse_order:
                variants.reverse()
            manifest['neural_summary'] = []
            for use_pruning in variants:
                result = neural_run(examples, wrapper, args.output, use_pruning, threshold, args.batch_size)
                manifest['neural_summary'].append(result)
                manifest['completed_runs'].append('pruned' if use_pruning else 'unpruned')
                write_json(args.output / 'manifest.json', manifest)
        manifest.update(status='completed', full_dataset_run=len(examples) == full_count)
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
