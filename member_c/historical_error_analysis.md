# Member C error analysis: historical draft

NOT A FINAL SUBMISSION. This draft examines existing cached predictions from the unchanged implementation. Fresh full-run reproduction and group review remain required before submitting the written report.

## Provenance and evaluation

The source is results/entailment_pruning_005_scores.json, SHA-256 4b43b0828f8fb8d7f39df5360965fee8ddc404934e1415935268596941ca3d20. It stores 221 claims for MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, CPU, with pruning threshold 0.05. We apply decision threshold 0.38 to its recorded maximum probabilities; IR is converted to NS exactly as in the supplied driver.

Cached predictions yield 187/221 correct (84.615%): TP=88, TN=99, FP=10, FN=24. We inspect the first 10 false positives and first 10 false negatives in driver order, satisfying the requested sample size for that historical configuration; the remaining 14 false negatives are not part of these category statistics.

The scores are internally consistent, but they are historical evidence. Windows CRLF normalization explains the driver/data signature differences; the source signature also matches the old entailment default of 0.5. This contribution leaves the existing sentence cleaning and all submitted logic unchanged. A fresh full run is still required to verify reproducibility in the current environment. No cache signature or gold annotation was changed.

## Four primary error categories

### A: Role or entity-type inference beyond an explicit statement

The prediction relies on generalising an occupation, activity, location or entity mention to another role, type or broader claim. Such inferences may be plausible, but they disagree with the supplied NS label and do not establish the dataset's intended support criterion.

### B: Relevance-label or retrieved-evidence mismatch

The binary evaluation and the retrieved text appear to assess different things: an IR fact is scored as NS, or the text supports or disputes a claim differently from its gold label. Labels are retained unchanged; these observations do not prove that an annotation is wrong.

### C: Bibliographic attribution across lists and context

Support depends on connecting a work or series title in a bibliography or descriptive list to the author identified elsewhere in the article. A sentence-level premise may not state that author-work relation explicitly.

### D: Subject and relation expressed indirectly across wording or context

The relevant relation is expressed through a pronoun, name variant, paraphrase or contextual detail rather than a close single-sentence match to the claim. The error is associated with that evidence pattern; its precise cause cannot be established from maximum probabilities alone.

## Aggregate statistics for the examined sample

| Category | FP | FN | Total |
| --- | --- | --- | --- |
| A: Role or entity-type inference beyond an explicit statement | 8 | 0 | 8 |
| B: Relevance-label or retrieved-evidence mismatch | 2 | 3 | 5 |
| C: Bibliographic attribution across lists and context | 0 | 4 | 4 |
| D: Subject and relation expressed indirectly across wording or context | 0 | 3 | 3 |
| Total examined | 10 | 10 | 20 |
Each examined error has exactly one primary category, so the totals reconcile to 20. Categories describe observed evidence patterns, not proven internal causes: the historical cache stores one maximum score per claim and no winning model span or complete three-class probabilities. Some NS labels conflict with plausible inferences; all original labels are preserved.

## Three detailed examples

### Example 1: driver example 2 (FP)

Fact: Marianne McAndrew is a singer.

Ground truth: NS; prediction: S; maximum cached entailment probability: 0.894920.

Primary error type: Role or entity-type inference beyond an explicit statement

Retrieved evidence excerpt: but McAndrew did not do her own singing in the film

The passages describe McAndrew as an actress and explicitly state that she did not do her own singing in the cited film. Predicting the separate occupation singer fits role extrapolation despite this caveat; maximum-score aggregation can accept one apparently supportive span without reconciling opposing evidence, although the cached file does not identify the winning span.

### Example 2: driver example 141 (FN)

Fact: She was born on July 18, 1987.

Ground truth: S; prediction: NS; maximum cached entailment probability: 0.000274.

Primary error type: Relevance-label or retrieved-evidence mismatch

Retrieved evidence excerpt: born in Maracaibo, Venezuela on 14 August 1987

The gold-supported claim gives 18 July 1987, but the supplied Jessica Barboza passage gives 14 August 1987. The low entailment score is compatible with rejecting that conflicting date, although it is still an FN under the fixed evaluation labels; the original labels are not changed.

### Example 3: driver example 79 (FN)

Fact: The Deep is a novel by Katsu.

Ground truth: S; prediction: NS; maximum cached entailment probability: 0.006295.

Primary error type: Bibliographic attribution across lists and context

Retrieved evidence excerpt: "The Deep", Transworld, March 2020

The Deep appears under the article's standalone-novels heading, and a separate passage calls it a finalist for novel awards. The author-work relation is distributed across article context, a list and another sentence rather than stated with Katsu in one direct premise, which fits bibliographic attribution failure.

## Member C allocation

Using the existing balanced team allocation, Member C receives examples 109, 122 and 124 (FP), and 79 and 80 (FN). All five were examined against their supplied retrieved passages below. Detailed example 3 is example 79. This is an AI-assisted draft prepared after Manura reported taking over the work; it is not evidence that Sachinthana completed it or that a human peer accepted it.

## Interpretation and limitations

Eight examined FPs involve role/type generalisation, five errors involve relevance or apparent label/evidence mismatches, four FNs involve bibliography attribution, and three FNs involve indirectly expressed subjects or relations. These counts describe this selected sample only; neither population error rates nor causal effects of pruning can be estimated from them.

Sentence-only evidence may lose article-level attribution and pronoun context. Maximum-score aggregation also accepts one strong support score without reconciling competing spans. These are limitations to explain, not measured causal conclusions: the historical cache cannot show whether a particular decisive sentence was pruned or which sentence determined a score.

The existing implementation retains dataset sentence markers, and an audit records a sentence-boundary/cleaning issue without changing the implementation. No fresh neural accuracy or pruning runtime benefit has been measured here because model access is denied. Tuning and reporting thresholds on the same development data is allowed by the brief but is not an independent test-set generalisation estimate.

## Evidence ledger: all 20 examined cached errors

### ID 2: FP, category A

Fact: Marianne McAndrew is a singer.

Original label NS; evaluated gold NS; predicted S; score 0.894920.

Evidence excerpt: but McAndrew did not do her own singing in the film

The passages describe McAndrew as an actress and explicitly state that she did not do her own singing in the cited film. Predicting the separate occupation singer fits role extrapolation despite this caveat; maximum-score aggregation can accept one apparently supportive span without reconciling opposing evidence, although the cached file does not identify the winning span.

### ID 5: FP, category B

Fact: Cleveland is in Ohio.

Original label IR; evaluated gold NS; predicted S; score 0.993614.

Evidence excerpt: Trivisonno was born in East Cleveland, Ohio

The original label is IR, which the driver converts to NS even though a geographic statement can receive strong entailment support. The retrieved passage mentions East Cleveland, Ohio; the counted error therefore involves relevance and geographic inference, not a demonstrated false real-world location.

### ID 8: FN, category B

Fact: Gerhard Fischer is best known for inventing a metal detector.

Original label S; evaluated gold S; predicted NS; score 0.003193.

Evidence excerpt: He contributed to the development and popularity of the hand held metal detector.

The gold label is S, but the evidence says contributed to development and elsewhere disputes who invented the handheld detector. This differs from the stronger best known for inventing formulation, so the FN is recorded as a label/evidence mismatch rather than proof that the model ignored unequivocal support.

### ID 9: FN, category B

Fact: Gerhard Fischer invented the first handheld, battery-operated metal detector.

Original label S; evaluated gold S; predicted NS; score 0.004529.

Evidence excerpt: the hand-held metal detector is disputed

The supplied text disputes the inventor and discusses patenting, while the claim adds first and battery-operated. Those qualifications are not explicitly established by the retrieved passages, although the fixed gold label is S; the discrepancy is preserved and not repaired by relabelling.

### ID 34: FP, category A

Fact: Tusa has been involved in various cultural initiatives.

Original label NS; evaluated gold NS; predicted S; score 0.403498.

Evidence excerpt: Tusa promoted excavations in the archaeological sites of Soluntum, Segesta, Selinunte, Motya, Marsala.

The text describes archaeological excavations and promotion of a magazine, while the claim summarises these as various cultural initiatives. This is a broad activity-level generalisation rather than the same explicit proposition, and it is counted as FP against the supplied NS label.

### ID 40: FP, category A

Fact: Jean Daullé was a printmaker.

Original label NS; evaluated gold NS; predicted S; score 0.925237.

Evidence excerpt: was a French engraver.

The source identifies Daullé as an engraver, whereas the claim uses the broader occupational term printmaker. That type inference is semantically plausible, so the FP should not be described as an obvious factual hallucination; it remains a mismatch with the supplied NS label.

### ID 41: FN, category D

Fact: Jean Daullé was known for his portraits.

Original label S; evaluated gold S; predicted NS; score 0.145188.

Evidence excerpt: He engraved several portraits and plates of historical and other subjects

The passage describes portrait engraving through the pronoun He, rather than saying directly that Jean Daullé was known for his portraits. Subject resolution and the change from producing portraits to being known for them are contextual and relational gaps associated with this FN.

### ID 77: FN, category C

Fact: The Taker trilogy is a novel by Katsu.

Original label S; evaluated gold S; predicted NS; score 0.017982.

Evidence excerpt: "The Taker Trilogy" tells the story of a young woman

The evidence describes The Taker Trilogy and lists its component titles under Katsu's bibliography instead of stating the claim as a direct author-work sentence. The claim also calls a trilogy a novel, so the fixed S label should not be presented as unambiguous textual entailment.

### ID 78: FN, category C

Fact: The Hunger is a novel by Katsu.

Original label S; evaluated gold S; predicted NS; score 0.006046.

Evidence excerpt: Bibliography.:Standalone novels. - "The Hunger", Transworld, March 2018

The Hunger occurs in a standalone-novels bibliography within the Alma Katsu article, but the list entry does not repeat Katsu's name. Interpreting the title as her novel requires list-heading and article context, making this a bibliographic-attribution FN.

### ID 79: FN, category C - allocated to Member C

Fact: The Deep is a novel by Katsu.

Original label S; evaluated gold S; predicted NS; score 0.006295.

Evidence excerpt: "The Deep", Transworld, March 2020

The Deep appears under the article's standalone-novels heading, and a separate passage calls it a finalist for novel awards. The author-work relation is distributed across article context, a list and another sentence rather than stated with Katsu in one direct premise, which fits bibliographic attribution failure.

### ID 80: FN, category C - allocated to Member C

Fact: Red Widow is a novel by Katsu.

Original label S; evaluated gold S; predicted NS; score 0.007195.

Evidence excerpt: "Red Widow", G.P. Putnam's Sons, March 2021

Red Widow is listed under standalone novels in the Alma Katsu bibliography. The entry supplies a title, publisher and date without repeating the author, so a sentence-level classifier must recover author attribution from the surrounding article context.

### ID 99: FN, category D

Fact: He served for many years as a pastor.

Original label S; evaluated gold S; predicted NS; score 0.034975.

Evidence excerpt: Between 1986 and 1989, Malcolm a pastor of the Jerusalem Christian Assembly.

The retrieved text expresses the pastoral role in a grammatically incomplete sentence and gives dates, while the claim uses He and the vague duration many years. Resolving the subject and relating the date interval to that duration requires contextual interpretation; the gold S label is retained without claiming an exact causal diagnosis.

### ID 105: FP, category A

Fact: She has worked with a number of organizations.

Original label NS; evaluated gold NS; predicted S; score 0.617346.

Evidence excerpt: She also delivers awareness raising sessions to doctors and the Police

The passage lists schools, doctors, police and mediator/educator activities, but the claim compresses this into worked with a number of organizations. The generalisation is plausible, yet the benchmark label is NS, so this FP is an activity-to-broader-role inference rather than an established false biography detail.

### ID 109: FP, category A - allocated to Member C

Fact: Nisar Ahmad Kakru was Kashmiri.

Original label NS; evaluated gold NS; predicted S; score 0.992450.

Evidence excerpt: born 26 October 1949 Baramulla, Kashmir, India

The evidence gives Kakru's birthplace in Kashmir, whereas Kashmiri can denote a regional identity or ethnicity. Moving from birthplace to that identity is an attribute inference; the high cached probability does not resolve the intended interpretation or override the supplied NS label.

### ID 121: FN, category D

Fact: Don Featherstone created the plastic pink flamingo lawn ornament.

Original label S; evaluated gold S; predicted NS; score 0.027660.

Evidence excerpt: was an American artist most widely known for his 1957 creation of the plastic pink flamingo

The passage supports creation using Donald Featherstone, while the claim uses Don Featherstone and adds lawn ornament; lawn context occurs elsewhere. This requires a name-variant match and a relation paraphrase with contextual detail, despite the clearly relevant retrieved evidence.

### ID 122: FP, category A - allocated to Member C

Fact: He has worked as a researcher at various organizations.

Original label NS; evaluated gold NS; predicted S; score 0.746819.

Evidence excerpt: he directed the BYU Biomedical Imaging Lab, and performed research on scanning impedance imaging.

The evidence explicitly reports research at BYU and separately lists several career roles and organisations. The claim expands this into working as a researcher at various organizations, a stronger multi-organisation role claim than that single research statement establishes.

### ID 124: FP, category A - allocated to Member C

Fact: Mayo Clinic is an organization.

Original label NS; evaluated gold NS; predicted S; score 0.781372.

Evidence excerpt: Oliphant has a Ph.D. in Biomedical Engineering from the Mayo Clinic

The passage mentions Mayo Clinic as a degree-granting institution without directly asserting the entity type organization. Inferring that type is reasonable in ordinary language, so this is a type-inference disagreement with the supplied NS annotation rather than evidence that the real-world claim is false.

### ID 133: FP, category B

Fact: He served as a sniper.

Original label NS; evaluated gold NS; predicted S; score 0.997330.

Evidence excerpt: is a Canadian former military sniper

The source explicitly identifies Rob Furlong as a former military sniper, which supports the claim if He refers to Furlong. Because the stored gold label is NS, this FP exposes a possible annotation or subject-context mismatch; it must not be explained as absence of sniper evidence.

### ID 141: FN, category B

Fact: She was born on July 18, 1987.

Original label S; evaluated gold S; predicted NS; score 0.000274.

Evidence excerpt: born in Maracaibo, Venezuela on 14 August 1987

The gold-supported claim gives 18 July 1987, but the supplied Jessica Barboza passage gives 14 August 1987. The low entailment score is compatible with rejecting that conflicting date, although it is still an FN under the fixed evaluation labels; the original labels are not changed.

### ID 168: FP, category A

Fact: Kerser is a songwriter.

Original label NS; evaluated gold NS; predicted S; score 0.945563.

Evidence excerpt: is an Australian rapper.

The source describes Kerser as a rapper and discusses recordings, albums and diss tracks, while the claim assigns the distinct role songwriter. The cached S decision fits a related-role inference, but the passages do not directly establish songwriting and the supplied label is NS.

## Sources and review status

Coursework requirements: Instructions/CW.pdf, especially pages 3-6 and 8-10. Team allocation: Instructions/october_team_execution_guide.pdf, pages 3 and 7. Model and labels: supplied repository code and data. Excerpts above are exact substrings of supplied retrieved passages, checked automatically. All interpretations were drafted with AI assistance; human review and viva preparation remain pending. Manura reports lecturer permission for AI use provided the team can explain and defend the work; retain the actual clarification with the group records.
