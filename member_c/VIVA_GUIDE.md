# Member C viva notes — unchanged implementation

Use these notes to explain the actual source and measured evidence, not to memorise claims of personal authorship. The role was assigned to Sachinthana and taken over by Manura with AI assistance; real peer review and student understanding must be demonstrated separately.

## Data and evaluation

The task is evidence-based support checking for human-extracted atomic facts from ChatGPT biographies. The loader produces 221 examples: 112 S, 93 NS and 16 IR. The unchanged driver maps IR to NS, so binary evaluation uses 112 S and 109 NS. The lexical sweep helper instead excludes IR; that denominator mismatch is reported without changing the helper.

## Passage aggregation and lexical scoring

Default preprocessing lowercases, NLTK-tokenizes, removes stopwords/non-alphanumeric tokens, and Porter-stems into sets. Recall is shared fact words divided by distinct fact words. With 4 fact words and 3 present in a passage, recall is 0.75, exceeding the default 0.65 threshold.

The classifier uses the highest passage score. One supporting passage can suffice; unrelated passages do not dilute it. Two passages each covering half the fact do not get added together. Empty evidence produces zero and NS under the positive default threshold. The audit checks these rules and the alternative lexical score formulas, but does not prove which scorer is best through a new tuning experiment.

## Sentence processing and the recorded issue

The existing entailment implementation collapses whitespace and calls NLTK sentence splitting. It does not remove dataset `<s>` and `</s>` markers. A marked two-sentence string can therefore reach the model as one tagged premise. A focused check fails, and 398 retained inputs contain markers in the structural audit. No correction was applied because teammates' completed code must remain unchanged.

Plain sentence boundaries and empty text checks pass. Sentence-level evidence can still lose pronoun or article context. Report that limitation without claiming it caused a particular historical error: the old cache has maximum scores only, not per-pair traces.

## Entailment and pruning

The premise is the evidence sentence; the hypothesis is the fact. The supplied wrapper reuses the loaded model, calls `eval()`, uses `torch.no_grad()`, limits token pairs to 512 tokens, applies softmax and reads the entailment index from the model configuration. Returned scores are Python floats rather than retained computation graphs. Actual checkpoint class order still needs to be recorded in the fresh model run.

The classifier takes the maximum entailment probability over retained sentences/passages and predicts S when it is at least 0.38. With no evidence it returns zero and NS. Low-overlap sentences are pruned below 0.05, which is separate from the lexical classification threshold 0.65. Case, punctuation and negation in model premises are not replaced by stemmed lexical tokens.

`evaluate_pruning.py` uses the existing implementation and compares pruning enabled/disabled under the same checkpoint, data, CPU/thread setting and threshold. A hook observes argmax from the same logits to compare discrete decisions with probability thresholding, without a second neural call. Synthetic tests confirm bookkeeping, not real-model accuracy. A recording substitute confirms filtering counts, not speedup or preservation of support.

## C's five historical errors

- **79, FN:** `The Deep is a novel by Katsu.` The title appears in the article's standalone-novels bibliography; author attribution depends on surrounding context. This is detailed example 3 in the draft.
- **80, FN:** `Red Widow is a novel by Katsu.` The bibliography again lists title, publisher and date without repeating the author.
- **109, FP:** `Nisar Ahmad Kakru was Kashmiri.` The passage gives a Kashmir birthplace; interpreting it as regional identity or ethnicity is a further attribute inference.
- **122, FP:** `He has worked as a researcher at various organizations.` Research at BYU is explicit, while other career roles are listed separately; the claim generalises the researcher role to multiple organisations.
- **124, FP:** `Mayo Clinic is an organization.` That entity-type inference is plausible, but the supplied label is NS. Do not describe that label as proof that the real-world proposition is false.

These are the current historical sample, not verified fresh-run errors. Read all retrieved passages and all three detailed examples in the PDF. Maximum-score caches cannot prove which sentence was decisive or whether a specific supporting sentence was pruned.

## Results and completion status

The required lexical driver achieves 179/221 (80.995%). The existing-code audit has 19 passes and one failure. The evaluator's three synthetic tests pass. Historical cached entailment scores yield 187/221 (84.615%) at threshold 0.38, but fresh neural verification and real pruning timings are blocked by checkpoint access.

Four categories account for the selected 10 FP and 10 FN: 8 role/type inferences, 5 relevance/evidence mismatches, 4 bibliography-attribution cases, and 3 indirect subject/relation cases. Those counts describe the sample, not all errors. Keep gold labels fixed, distinguish apparent annotation/context tension from a proven annotation error, and distinguish a target accuracy from a measured result.

Development-set tuning/reporting is allowed by the brief but is not independent test-set generalisation. To finish, reproduce both required commands, verify the final error sample, review the final report against the code, and submit the final PDF plus `factcheck.py` separately. No result should be fabricated to close a checklist.
