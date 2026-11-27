"""Four-way retrieval experiment. Expected-term coverage is not faithfulness."""
import json
import time
import numpy as np
import engine


def run(progress=lambda text: None):
    questions = json.loads((engine.ROOT / 'questions.json').read_text())
    report = {'status':'running', 'corpus_digest':engine.corpus_digest(engine.corpus()),
              'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'runs':[], 'summary':[]}
    path = engine.ROOT / 'reports' / 'evaluation.json'
    for strategy in ('fixed', 'semantic'):
        for rerank in (False, True):
            group = []
            for case in questions:
                progress(f'{strategy} / rerank {rerank}: {len(report["runs"])+1} of {len(questions)*4}')
                start = time.monotonic()
                hits = engine.retrieve(case['question'], strategy, case['year'], rerank, live=True)
                terms = case['expected_terms']
                text = ' '.join(h['text'] for h in hits).lower()
                row = {'strategy':strategy, 'rerank':rerank, 'question':case['question'],
                    'seconds':round(time.monotonic()-start,3),
                    'expected_term_coverage':sum(t.lower() in text for t in terms)/len(terms) if terms else None,
                    'human_relevance_labels':None, 'human_faithfulness':None, 'hits':hits}
                report['runs'].append(row)
                group.append(row)
                engine.atomic_json(path, report)
            scored = [r['expected_term_coverage'] for r in group if r['expected_term_coverage'] is not None]
            report['summary'].append({'Chunking':strategy, 'Reranking':'On' if rerank else 'Off',
                'Numeric term coverage @5':round(float(np.mean(scored)),3) if scored else None,
                'Retrieval p95 (s)':round(float(np.percentile([r['seconds'] for r in group],95)),2),
                'Queries':len(group), 'Numeric questions':len(scored)})
    report['status'] = 'retrieval complete; human grading pending'
    engine.atomic_json(path, report)
    return report

if __name__ == '__main__':
    run(print)


def run_answers(progress=lambda text: None, strategy='semantic', rerank=True):
    """Evaluate end-to-end outcomes; keep automatic judgments distinct from human labels."""
    import workflow
    from services import ServiceError
    questions=json.loads((engine.ROOT/'questions.json').read_text())
    report={'status':'running','workflow_version':workflow.VERSION,'strategy':strategy,'rerank':rerank,
            'corpus_digest':engine.corpus_digest(engine.corpus()),'runs':[]}
    path=engine.ROOT/'reports'/'answer-evaluation.json'
    for i,case in enumerate(questions,1):
        progress(f'Checking full answers: {i}/{len(questions)}')
        try:
            result=workflow.research(case['question'],strategy,case['year'],rerank)
        except ServiceError as error:
            report['status']='failed'
            report['error']=str(error)
            engine.atomic_json(path,report)
            raise
        result['expected_answerable']=case['answerable']
        result['human_faithfulness']=None
        # An invalid reviewer response is not counted as a successful refusal.
        result['expected_outcome_matched']=(result['status']=='grounding_check_passed' if case['answerable']
                                            else result['status']=='insufficient_evidence')
        report['runs'].append(result)
        engine.atomic_json(path,report)
    report['status']='complete; human faithfulness grading pending'
    report['summary']={'Questions':len(questions),
        'Expected outcome matches':sum(r['expected_outcome_matched'] for r in report['runs']),
        'Verification failures':sum(r['status']=='verification_failed' for r in report['runs']),
        'End-to-end p95 seconds':round(float(np.percentile([r['seconds'] for r in report['runs']],95)),2)}
    engine.atomic_json(path,report)
    return report
