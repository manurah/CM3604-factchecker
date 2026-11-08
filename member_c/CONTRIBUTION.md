# Member C contribution record

Assigned role: Sachinthana (C). Takeover reported by: Manura. Preparation: Manura with AI assistance. Human peer acceptance: pending. The official brief defines group deliverables; the T-numbers below come from the team's execution guide.

| Task | Evidence in this contribution | What remains |
|---|---|---|
| T03: inspect data/IR/evidence | Complete machine audit: 31 biographies, 221 loaded examples, no missing-fact warnings; original labels and IR-to-NS mapping recorded. | Student teach-back and human review. |
| T07: aggregation/segmentation design | Existing maximum aggregation and sentence processing traced and documented. | Group acceptance; implementation issue recorded without a source change. |
| T11: passage aggregation | Existing maximum rule verified with unrelated, supporting, reordered and empty passages. | No new implementation or authorship claim; preserve teammates' code. |
| T15: check lexical scoring | Controlled recall/Jaccard/F1/ROUGE arithmetic and boundary cases checked. | Human peer acceptance. |
| T19: sentence splitting/cleaning | Reproduced failure on dataset `<s>` markers; 398 retained model inputs contain markers under existing processing. | Code issue reported to team; no patch authorised under the preservation constraint. |
| T23: pruning comparison | Separate evaluator and three synthetic-backend tests prepared; structural counts reproduced. | Actual pruning-on/off accuracy/runtime comparison needs checkpoint access. |
| T27: final entailment reproduction | Historical score/data alignment and CSV arithmetic verified. | Fresh full model run is blocked; historical arithmetic is not a reproduction. |
| T31: review 3 FP/2 FN and D's records | Five C cases examined against retrieved passages with original labels preserved. | Confirm errors against the final run, and perform actual human cross-review of D's records. |
| T35: analysis and example 3 | Historical draft supplies four categories, aggregate counts and three detailed examples, including C's example 3. | Final-run confirmation and group review before submission. |
| T39: package audit | All original tracked files fingerprinted; original driver/data/dependencies/results preserved. | Final PDF, fresh entailment verification and human approval remain outstanding. |

## Measured evidence and limitations

- Required lexical driver: 179/221 correct (80.995%); original code unchanged.
- Existing-code audit: 20 checks, 19 passed, 1 failed, 0 skipped. The failure exposes an existing markup/boundary issue; it is retained in the audit rather than hidden.
- Evaluator bookkeeping tests: 3 passed using a synthetic PyTorch backend, not the DeBERTa checkpoint.
- Existing pruning structure: 5,570 candidate sentence occurrences, 3,951 pruned, 1,619 retained. These are recording-substitute counts, not measured model runtime or accuracy preservation.
- Historical cached score arithmetic at threshold 0.38: 187/221 (84.615%), TP=88/TN=99/FP=10/FN=24. Exactly 10 FP and the first 10 FN in driver order were examined; the other 14 FN are outside the category sample.
- Four primary categories in that sample: role/type inference 8; relevance/evidence mismatch 5; bibliography attribution 4; indirect subject/relation expression 3. Counts sum to 20 and are not population error rates.
- The cached raw hashes differ because of Windows CRLF endings and the historical default threshold 0.5. Those differences were investigated and reproduced. No hashes, labels or scores were rewritten.
- No performance gain from changing preprocessing is claimed: the existing preprocessing remains unchanged. No invented timings, model runs, authorship or peer-review signatures are included.

## Why some historical errors require careful wording

An evaluated error is disagreement with the supplied label, not proof of a false real-world claim. For example, the source gives a different birth date from the gold-supported date in example 141, and explicitly mentions a sniper role in example 133 despite its NS label. The draft retains those labels and discusses the apparent evidence/context mismatch without declaring the labels wrong.

## Scope boundary

This contribution adds only the `member_c/` folder. Issues found in existing implementations are documented for the team. It does not alter, replace, reattribute or claim ownership of work already completed by Sandali or Dilshan. Earlier proposed classifier patches are outside the repository and are not included in this commit.
