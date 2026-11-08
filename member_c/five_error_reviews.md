# Member C: five historical error reviews

Historical cached predictions at threshold 0.38. Prepared for the Member C scope taken over by Manura with AI assistance. Fresh model reproduction and human review are pending.

## ID 79: FN

**Fact:** The Deep is a novel by Katsu.

**Gold:** S; **prediction:** NS; **score:** 0.006295.

**Category:** Bibliographic attribution across lists and context

**Retrieved evidence:** "The Deep", Transworld, March 2020

The Deep appears under the article's standalone-novels heading, and a separate passage calls it a finalist for novel awards. The author-work relation is distributed across article context, a list and another sentence rather than stated with Katsu in one direct premise, which fits bibliographic attribution failure.

## ID 80: FN

**Fact:** Red Widow is a novel by Katsu.

**Gold:** S; **prediction:** NS; **score:** 0.007195.

**Category:** Bibliographic attribution across lists and context

**Retrieved evidence:** "Red Widow", G.P. Putnam's Sons, March 2021

Red Widow is listed under standalone novels in the Alma Katsu bibliography. The entry supplies a title, publisher and date without repeating the author, so a sentence-level classifier must recover author attribution from the surrounding article context.

## ID 109: FP

**Fact:** Nisar Ahmad Kakru was Kashmiri.

**Gold:** NS; **prediction:** S; **score:** 0.992450.

**Category:** Role or entity-type inference beyond an explicit statement

**Retrieved evidence:** born 26 October 1949 Baramulla, Kashmir, India

The evidence gives Kakru's birthplace in Kashmir, whereas Kashmiri can denote a regional identity or ethnicity. Moving from birthplace to that identity is an attribute inference; the high cached probability does not resolve the intended interpretation or override the supplied NS label.

## ID 122: FP

**Fact:** He has worked as a researcher at various organizations.

**Gold:** NS; **prediction:** S; **score:** 0.746819.

**Category:** Role or entity-type inference beyond an explicit statement

**Retrieved evidence:** he directed the BYU Biomedical Imaging Lab, and performed research on scanning impedance imaging.

The evidence explicitly reports research at BYU and separately lists several career roles and organisations. The claim expands this into working as a researcher at various organizations, a stronger multi-organisation role claim than that single research statement establishes.

## ID 124: FP

**Fact:** Mayo Clinic is an organization.

**Gold:** NS; **prediction:** S; **score:** 0.781372.

**Category:** Role or entity-type inference beyond an explicit statement

**Retrieved evidence:** Oliphant has a Ph.D. in Biomedical Engineering from the Mayo Clinic

The passage mentions Mayo Clinic as a degree-granting institution without directly asserting the entity type organization. Inferring that type is reasonable in ordinary language, so this is a type-inference disagreement with the supplied NS annotation rather than evidence that the real-world claim is false.
