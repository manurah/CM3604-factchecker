"""Member D: binary metrics, cached-score threshold sweeps and error selection.

Gold labels are used only here for evaluation, never by either classifier.
IR maps to NS, exactly as in the supplied coursework driver.
"""
import math
from decimal import Decimal


def binary_label(label):
    if label not in ('S', 'NS', 'IR'):
        raise ValueError(f'Unknown gold label: {label!r}')
    return 'NS' if label == 'IR' else label


def binary_metrics(labels, predictions):
    labels, predictions = list(labels), list(predictions)
    if len(labels) != len(predictions):
        raise ValueError('Gold and prediction lengths differ.')
    counts = dict(true_positives=0, true_negatives=0, false_positives=0, false_negatives=0)
    for gold, prediction in zip(labels, predictions):
        gold = binary_label(gold)
        if prediction not in ('S', 'NS'):
            raise ValueError(f'Invalid prediction: {prediction!r}')
        key = ('true_positives' if prediction == 'S' else 'false_negatives') if gold == 'S' else (
            'false_positives' if prediction == 'S' else 'true_negatives')
        counts[key] += 1
    correct = counts['true_positives'] + counts['true_negatives']
    return dict(counts, total=len(labels), correct=correct,
                accuracy=correct / len(labels) if labels else None)


def threshold_sweep(labels, scores, thresholds=None, reference=0.65):
    """Score once, evaluate many cutoffs without rerunning a classifier.

    Integer-derived hundredths avoid accumulating floating-point step errors.
    Ties prefer the cutoff closest to a declared reference, then the lower one.
    Callers using entailment should exclude zero, which would accept no evidence.
    """
    labels, scores = list(labels), list(scores)
    if not labels or len(labels) != len(scores):
        raise ValueError('A nonempty set of aligned labels and scores is required.')
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError('Support scores must be finite probabilities/overlaps in [0, 1].')
    cutoffs = list(thresholds) if thresholds is not None else [step / 100 for step in range(101)]
    if not cutoffs or any(not math.isfinite(t) or not 0 <= t <= 1 for t in cutoffs):
        raise ValueError('Thresholds must be a nonempty set of finite cutoffs in [0, 1].')
    if not math.isfinite(reference) or not 0 <= reference <= 1:
        raise ValueError('Reference threshold must be finite and in [0, 1].')
    rows = [dict(threshold=t, **binary_metrics(labels, ['S' if s >= t else 'NS' for s in scores]))
            for t in sorted(set(cutoffs))]
    best = max(rows, key=lambda row: (
        row['correct'], -abs(Decimal(str(row['threshold'])) - Decimal(str(reference))), -row['threshold']))
    return rows, best


def select_errors(records, per_type=10):
    """Select actual errors in driver-ID order; never invent missing examples."""
    if not isinstance(per_type, int) or isinstance(per_type, bool) or per_type < 0:
        raise ValueError('per_type must be a nonnegative integer.')
    records = list(records)
    identifiers = [r['example_id'] for r in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Duplicate example IDs would double-count errors.')
    selected, counts = [], {'FP': 0, 'FN': 0}
    for record in sorted(records, key=lambda row: row['example_id']):
        gold = binary_label(record['original_label'])
        prediction = record['prediction']
        if prediction not in ('S', 'NS'):
            raise ValueError('Error selection needs valid S/NS predictions.')
        if 'gold' in record and record['gold'] != gold:
            raise ValueError('Stored binary gold differs from the original label.')
        if prediction != gold:
            kind = 'FP' if prediction == 'S' else 'FN'
            if counts[kind] < per_type:
                selected.append(dict(record, gold=gold, error_type=kind))
                counts[kind] += 1
    return selected
