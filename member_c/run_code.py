"""Run Member C's opt-in components while leaving the supplied driver unchanged.

From the repository: python -m member_c.run_code --mode word_overlap
For neural checking: python -m member_c.run_code --mode entailment [--cuda]
The latter needs the same public checkpoint/cache as the supplied driver.
"""
import argparse
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factcheck import EntailmentModel
from factchecking_main import read_fact_examples, read_passages, predict_two_classes
from member_c.evidence_processing import (
    PassageAggregatingWordFactChecker,
    SentenceAwareEntailmentFactChecker,
)

PROJECT = Path(__file__).resolve().parents[1]
MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['word_overlap', 'entailment'])
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--labels_path', type=Path, default=PROJECT / 'data/dev_labeled_ChatGPT.jsonl')
    parser.add_argument('--passages_path', type=Path, default=PROJECT / 'data/passages_bm25_ChatGPT_humfacts.jsonl')
    args = parser.parse_args()
    examples = read_fact_examples(str(args.labels_path), read_passages(str(args.passages_path)))
    if not examples:
        raise ValueError('No examples were loaded.')
    print(f'Read {len(examples)} examples; Member C mode: {args.mode}')
    if args.mode == 'word_overlap':
        checker = PassageAggregatingWordFactChecker()
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        if args.cuda:
            model.to('cuda')
        checker = SentenceAwareEntailmentFactChecker(EntailmentModel(model, tokenizer, args.cuda))
    predict_two_classes(examples, checker)


if __name__ == '__main__':
    main()
