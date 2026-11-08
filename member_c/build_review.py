"""Build the explicitly historical review draft from inspected annotations.

Requires reportlab only to render the optional PDF; it is not an application
dependency. Never changes the original score cache or its gold labels.
"""
from collections import Counter
from html import escape
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEST = PROJECT / 'member_c'


def build_sections():
    source = PROJECT / 'results/entailment_pruning_005_scores.json'
    saved = json.loads(source.read_text())
    notes = json.loads((DEST / 'historical_annotations.json').read_text())
    threshold = notes['threshold']
    by_id = {r['example_id']: r for r in saved['records']}
    all_errors = [r for r in saved['records'] if ('S' if r['score'] >= threshold else 'NS') != r['gold']]
    fp = [r for r in all_errors if r['gold'] == 'NS']
    fn = [r for r in all_errors if r['gold'] == 'S']
    selected = fp[:10] + fn[:10]
    assert len(notes['records']) == len(selected) == 20
    assert {r['example_id'] for r in notes['records']} == {r['example_id'] for r in selected}
    counts = Counter()
    reviewed = []
    for annotation in notes['records']:
        record = by_id[annotation['example_id']]
        assert any(annotation['quote'] in p['text'] for p in record['passages']), f'Quote not found: {record["example_id"]}'
        category = annotation['category']
        assert category in notes['categories']
        error_type = 'FP' if record['gold'] == 'NS' else 'FN'
        counts[category, error_type] += 1
        reviewed.append(dict(record, **{k: v for k, v in annotation.items() if k != 'example_id'},
                             prediction='S' if error_type == 'FP' else 'NS', error_type=error_type,
                             human_review_accepted=False))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    sections = [
        ('title', 'Member C error analysis: historical draft'),
        ('paragraph', 'NOT A FINAL SUBMISSION. This draft examines existing cached predictions from the unchanged implementation. Fresh full-run reproduction and group review remain required before submitting the written report.'),
        ('heading', 'Provenance and evaluation'),
        ('paragraph', f'The source is results/entailment_pruning_005_scores.json, SHA-256 {source_sha}. It stores 221 claims for MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, CPU, with pruning threshold 0.05. We apply decision threshold {threshold:.2f} to its recorded maximum probabilities; IR is converted to NS exactly as in the supplied driver.'),
        ('paragraph', 'Cached predictions yield 187/221 correct (84.615%): TP=88, TN=99, FP=10, FN=24. We inspect the first 10 false positives and first 10 false negatives in driver order, satisfying the requested sample size for that historical configuration; the remaining 14 false negatives are not part of these category statistics.'),
        ('paragraph', 'The scores are internally consistent, but they are historical evidence. Windows CRLF normalization explains the driver/data signature differences; the source signature also matches the old entailment default of 0.5. This contribution leaves the existing sentence cleaning and all submitted logic unchanged. A fresh full run is still required to verify reproducibility in the current environment. No cache signature or gold annotation was changed.'),
        ('heading', 'Four primary error categories'),
    ]
    for key, category in notes['categories'].items():
        sections.append(('subheading', key + ': ' + category['name']))
        sections.append(('paragraph', category['definition']))
    rows = [['Category', 'FP', 'FN', 'Total']]
    for key, category in notes['categories'].items():
        rows.append([key + ': ' + category['name'], str(counts[key, 'FP']), str(counts[key, 'FN']), str(counts[key, 'FP'] + counts[key, 'FN'])])
    rows.append(['Total examined', '10', '10', '20'])
    sections += [('heading', 'Aggregate statistics for the examined sample'), ('table', rows),
                 ('paragraph', 'Each examined error has exactly one primary category, so the totals reconcile to 20. Categories describe observed evidence patterns, not proven internal causes: the historical cache stores one maximum score per claim and no winning model span or complete three-class probabilities. Some NS labels conflict with plausible inferences; all original labels are preserved.'),
                 ('heading', 'Three detailed examples')]
    reviewed_by_id = {r['example_id']: r for r in reviewed}
    for number, example_id in enumerate((2, 141, 79), 1):
        r = reviewed_by_id[example_id]
        sections += [('subheading', f'Example {number}: driver example {example_id} ({r["error_type"]})'),
                     ('paragraph', 'Fact: ' + r['fact']),
                     ('paragraph', f'Ground truth: {r["gold"]}; prediction: {r["prediction"]}; maximum cached entailment probability: {r["score"]:.6f}.'),
                     ('paragraph', 'Primary error type: ' + notes['categories'][r['category']]['name']),
                     ('paragraph', 'Retrieved evidence excerpt: ' + r['quote']),
                     ('paragraph', r['explanation'])]
    sections += [
        ('heading', 'Member C allocation'),
        ('paragraph', 'Using the existing balanced team allocation, Member C receives examples 109, 122 and 124 (FP), and 79 and 80 (FN). All five were examined against their supplied retrieved passages below. Detailed example 3 is example 79. This is an AI-assisted draft prepared after Manura reported taking over the work; it is not evidence that Sachinthana completed it or that a human peer accepted it.'),
        ('heading', 'Interpretation and limitations'),
        ('paragraph', 'Eight examined FPs involve role/type generalisation, five errors involve relevance or apparent label/evidence mismatches, four FNs involve bibliography attribution, and three FNs involve indirectly expressed subjects or relations. These counts describe this selected sample only; neither population error rates nor causal effects of pruning can be estimated from them.'),
        ('paragraph', 'Sentence-only evidence may lose article-level attribution and pronoun context. Maximum-score aggregation also accepts one strong support score without reconciling competing spans. These are limitations to explain, not measured causal conclusions: the historical cache cannot show whether a particular decisive sentence was pruned or which sentence determined a score.'),
        ('paragraph', 'The existing implementation retains dataset sentence markers, and an audit records a sentence-boundary/cleaning issue without changing the implementation. No fresh neural accuracy or pruning runtime benefit has been measured here because model access is denied. Tuning and reporting thresholds on the same development data is allowed by the brief but is not an independent test-set generalisation estimate.'),
        ('heading', 'Evidence ledger: all 20 examined cached errors'),
    ]
    for r in sorted(reviewed, key=lambda r: r['example_id']):
        member = ' - allocated to Member C' if r['example_id'] in (109, 122, 124, 79, 80) else ''
        sections += [('subheading', f'ID {r["example_id"]}: {r["error_type"]}, category {r["category"]}{member}'),
                     ('paragraph', 'Fact: ' + r['fact']),
                     ('paragraph', f'Original label {r["original_label"]}; evaluated gold {r["gold"]}; predicted {r["prediction"]}; score {r["score"]:.6f}.'),
                     ('paragraph', 'Evidence excerpt: ' + r['quote']),
                     ('paragraph', r['explanation'])]
    sections += [('heading', 'Sources and review status'),
                 ('paragraph', 'Coursework requirements: Instructions/CW.pdf, especially pages 3-6 and 8-10. Team allocation: Instructions/october_team_execution_guide.pdf, pages 3 and 7. Model and labels: supplied repository code and data. Excerpts above are exact substrings of supplied retrieved passages, checked automatically. All interpretations were drafted with AI assistance; human review and viva preparation remain pending. Manura reports lecturer permission for AI use provided the team can explain and defend the work; retain the actual clarification with the group records.')]
    ledger = {'status': notes['status'], 'source_sha256': source_sha,
              'fresh_model_run_verified': False, 'threshold': threshold,
              'categories': notes['categories'], 'records': reviewed}
    return sections, ledger


def render_pdf(sections, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    fonts = Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('ReviewSans', str(fonts / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('ReviewSansBold', str(fonts / 'DejaVuSans-Bold.ttf')))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = 'ReviewSans'
    styles['Normal'].fontSize = 9
    styles['Normal'].leading = 12.5
    styles['Normal'].spaceAfter = 6
    for name in ('Title', 'Heading1', 'Heading2'):
        styles[name].fontName = 'ReviewSansBold'
    styles['Title'].fontSize = 18
    styles['Title'].leading = 23
    styles['Heading1'].fontSize = 12
    styles['Heading1'].leading = 17
    styles['Heading2'].fontSize = 10
    styles['Heading2'].leading = 14
    flow = []
    for kind, value in sections:
        if kind == 'table':
            table = Table([[Paragraph(escape(cell), styles['Normal']) for cell in row] for row in value],
                          colWidths=[330, 40, 40, 45], repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6edf2')),
                                       ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                       ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
                                       ('LEFTPADDING', (0, 0), (-1, -1), 6),
                                       ('RIGHTPADDING', (0, 0), (-1, -1), 6)]))
            flow.extend([table, Spacer(1, 10)])
        else:
            style = {'title': 'Title', 'heading': 'Heading1', 'subheading': 'Heading2', 'paragraph': 'Normal'}[kind]
            flow.append(Paragraph(escape(value), styles[style]))
    def page_marks(canvas, document):
        canvas.saveState()
        canvas.setFont('ReviewSansBold', 7)
        canvas.setFillColor(colors.HexColor('#8d2020'))
        canvas.drawString(48, A4[1] - 31, 'HISTORICAL DRAFT - fresh full-run reproduction pending')
        canvas.setFont('ReviewSans', 7)
        canvas.setFillColor(colors.black)
        canvas.drawString(48, 25, 'AI-assisted review draft; human verification pending')
        canvas.drawRightString(A4[0] - 48, 25, str(document.page))
        canvas.restoreState()
    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=48, rightMargin=48,
                                 topMargin=52, bottomMargin=46, title='Member C historical error analysis - DRAFT')
    document.build(flow, onFirstPage=page_marks, onLaterPages=page_marks)


def main():
    sections, ledger = build_sections()
    markdown = []
    for kind, value in sections:
        if kind == 'table':
            markdown.append('| ' + ' | '.join(value[0]) + ' |')
            markdown.append('| ' + ' | '.join('---' for _ in value[0]) + ' |')
            markdown.extend('| ' + ' | '.join(row) + ' |' for row in value[1:])
        else:
            prefix = {'title': '# ', 'heading': '## ', 'subheading': '### ', 'paragraph': ''}[kind]
            markdown.append(prefix + value)
        if kind != 'table':
            markdown.append('')
    (DEST / 'historical_error_analysis.md').write_text('\n'.join(markdown).rstrip() + '\n')
    (DEST / 'historical_reviewed_errors.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + '\n')
    c_ids = {79, 80, 109, 122, 124}
    five = ['# Member C: five historical error reviews', '',
            'Historical cached predictions at threshold 0.38. Prepared for the Member C scope taken over by Manura with AI assistance. Fresh model reproduction and human review are pending.', '']
    for record in ledger['records']:
        if record['example_id'] in c_ids:
            five.extend([
                f'## ID {record["example_id"]}: {record["error_type"]}', '',
                '**Fact:** ' + record['fact'], '',
                f'**Gold:** {record["gold"]}; **prediction:** {record["prediction"]}; **score:** {record["score"]:.6f}.', '',
                '**Category:** ' + ledger['categories'][record['category']]['name'], '',
                '**Retrieved evidence:** ' + record['quote'], '',
                record['explanation'], '',
            ])
    (DEST / 'five_error_reviews.md').write_text('\n'.join(five).rstrip() + '\n')
    render_pdf(sections, DEST / 'historical_error_analysis_DRAFT.pdf')
    print('Built historical Markdown/PDF and evidence ledger; verified 20 source quotes and exact 10 FP/10 FN selection.')


if __name__ == '__main__':
    main()
