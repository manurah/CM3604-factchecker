# Member C — Sachinthana's assigned scope

This folder makes the contribution reviewable in Git while preserving all existing project files.

**Attribution:** this is Sachinthana's assigned role, taken over by **Manura with AI assistance**, as discussed by the group. It is not evidence that Sachinthana personally authored these files or that another member accepted them in peer review. Manura reports lecturer permission for AI use provided the students can explain and defend their work during the viva.

## What this contribution contains

| Work | Where to see it | Status |
|---|---|---|
| Data, label and evidence inspection | `results/baseline-audit/audit.json` | Verified: 221 examples; 112 S, 93 NS, 16 IR; required driver maps IR to NS. |
| Lexical scoring and aggregation checks | `check_existing.py`, `results/baseline-audit/behaviour-checks.txt` | 20 checks executed: 19 pass and 1 exposes an existing sentence-cleaning issue. Synthetic scores test behaviour, not model quality. |
| Sentence-processing review | `CONTRIBUTION.md`, audit results | Existing dataset-marker issue recorded. No fix applied to teammates' code. |
| Pruning on/off comparison tooling | `evaluate_pruning.py`, `tests/test_evaluator.py` | Three synthetic-backend tool tests pass. Actual neural comparison remains blocked by model access. |
| C's 3 FP and 2 FN reviews | [Five error reviews](five_error_reviews.md) | Historical cached examples 109, 122, 124 (FP) and 79, 80 (FN); exact evidence quotes checked. |
| Analysis section and detailed example 3 | [Analysis draft](historical_error_analysis.md), [PDF draft](historical_error_analysis_DRAFT.pdf) | Includes four category definitions, reconciled counts, three detailed examples and a 20-error evidence ledger supporting the group analysis. Historical draft; human review pending. |
| Understanding and explanation | [Viva notes](VIVA_GUIDE.md) | Describes the unchanged implementation and evidence limitations. |
| Protection of completed work | `protected_files_sha256.json` | Hashes of every original tracked file before this contribution. This change adds files only under `member_c/`. |

## What is deliberately unchanged

`factcheck.py`, `factchecking_main.py`, `threshold_sweep.py`, `evaluate_entailment.py`, `inspect_entailment.py`, README, requirements, data and the original result files are untouched. The required commands do not import this folder or depend on it. No classifier implementation is attributed to C by this commit.

The report and audit are contributions to **verification and analysis**, not an assertion that every task assigned to C is finished. See [CONTRIBUTION.md](CONTRIBUTION.md) for the task-by-task record and blockers.

## Run the checks

Use the repository's Python 3.10 environment and pinned dependencies. In this cloud instance:

```bash
cd /workspace/CM3604-factchecker
source .venv/bin/activate
export NLTK_DATA=/workspace/.cache/factchecker/nltk_data
export HF_HOME=/workspace/.cache/factchecker/huggingface
python member_c/verify_original_files.py
python -m unittest discover -s member_c/tests -v
python member_c/check_existing.py --output member_c/results/audit-next
```

Elsewhere, use your own checkout directory and the NLTK data location from the project setup. Both scripts resolve the repository relative to their own path; they do not depend on `/workspace`.

The tool tests should pass. **The existing-code audit currently exits 1**, retaining the finding that `<s>... </s><s>... </s>` text can reach the model with markers and merged boundaries. That failure is not disabled, counted as a pass, or fixed in this contribution. Each audit output directory must be new so evidence is not overwritten.

The full word-overlap command was run separately and passes with **179/221 (80.995%)**. Its original implementation is unchanged. The lexical sweep helper excludes IR and uses 205 examples, unlike the required driver; this is reported, not patched.

## Finish the real pruning comparison

Model access to `huggingface.co` currently returns proxy 403, and the checkpoint is absent from the configured cache. No fresh DeBERTa predictions or model-runtime comparison are claimed. Restore that access in environment settings, then run:

```bash
python member_c/evaluate_pruning.py --output member_c/results/pruning-run-01
```

The script uses the **existing** `EntailmentFactChecker` and inference wrapper. It compares pruning enabled/disabled with the same model, CPU threads, data and threshold, and records probability and argmax decisions from the same logits. A local copy of the exact checkpoint can be supplied with `--model /path/to/checkpoint --local-files-only`. Output directories must be new. `--limit` means a debug subset and must not be presented as full evaluation.

Only a completed manifest with `full_dataset_run: true` establishes a full comparison run. A failed manifest with no completed variants is not a successful run. Timings exclude checkpoint loading and result-file writes but include tokenization, Python instrumentation and existing garbage collection. Repeat with `--order unpruned-first` in a new directory before attributing small timing differences to pruning.

After the final configuration is confirmed, run both required commands, retain the final predictions, update the error sample if it differs from the historical one, and have the group review the analysis. C also reviews D's error records; that human review is not fabricated here. Submit the final PDF and `factcheck.py` separately, as CW.pdf requires. This historical draft is not labelled ready to submit.

## Rebuild the historical review

```bash
python3 member_c/build_review.py
```

This optional renderer uses ReportLab and DejaVu fonts, available in the current cloud Python environment. They are not new requirements for running the coursework classifiers. It reads the original cached scores and `historical_annotations.json`, verifies the exact 10 FP/10 FN selection and every evidence quote, and regenerates only files in this folder. It does not fetch a model or rewrite the original cache.

To inspect the Git contribution:

```bash
git log --oneline -- member_c/
git diff --stat origin/main...HEAD
```
