"""Live demo setup: fail honestly, save each stage, and never log secrets."""
import json
import re
import time
import engine
import workflow
from services import PineconeREST, ServiceError, completion

REPORT = engine.ROOT / 'reports' / 'demo-checks.json'


def run(progress=lambda message: None, rebuild=False):
    report = {'started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'status':'running', 'workflow_version':workflow.VERSION, 'checks':[], 'research':[]}
    REPORT.parent.mkdir(exist_ok=True)

    def stage(name, operation):
        progress(name)
        start = time.monotonic()
        try:
            value = operation()
        except Exception as error:
            detail = str(error) if isinstance(error, ServiceError) else f'{type(error).__name__}: check local dependencies and configuration.'
            report['checks'].append({'check':name, 'status':'failed', 'seconds':round(time.monotonic()-start,2), 'detail':detail})
            report['status'] = 'failed'
            engine.atomic_json(REPORT, report)
            raise ServiceError(detail) from None
        report['checks'].append({'check':name, 'status':'passed', 'seconds':round(time.monotonic()-start,2)})
        engine.atomic_json(REPORT, report)
        return value

    stage('Pinecone authentication', lambda: PineconeREST().control('/indexes'))

    def test_embedding():
        vectors = engine.embed(['Microsoft reported annual revenue.'])
        if len(vectors) != 1 or len(vectors[0]) != 1024:
            raise ServiceError('Expected one 1024-dimensional embedding.')
    stage('Pinecone embeddings · E5 / 1024 dimensions', test_embedding)

    def test_rerank():
        rows = PineconeREST().rerank('What is annual revenue?',
            ['Annual revenue is sales earned during the fiscal year.', 'A recipe for apple pie.'], top_n=2)
        if rows[0]['index'] != 0:
            raise ServiceError('Reranker did not select the relevant passage in the sanity check.')
    stage('Pinecone neural reranking', test_rerank)
    stage('Nebius generation', lambda: completion([{'role':'user','content':'Reply with exactly OK'}], max_tokens=16))
    documents = engine.corpus()
    if not documents:
        documents = stage('Download annual reports', engine.acquire)
    report['corpus_digest'] = engine.corpus_digest(documents)

    def prepare():
        if not rebuild:
            try:
                manifest = engine.load_manifest()
                if all((engine.DATA / f'{s}.json').exists() for s in ('fixed','semantic')):
                    return manifest
            except ServiceError:
                pass
        return engine.index_corpus(documents, progress)
    manifest = stage('Build / verify both retrieval indexes', prepare)
    report['counts'] = manifest['counts']

    for strategy in ('fixed','semantic'):
        for rerank in (False, True):
            def research():
                query = 'What was Microsoft total revenue in fiscal 2025?'
                start = time.monotonic()
                result = workflow.research(query, strategy, '2025', rerank, progress)
                report['research'].append(result)
                engine.atomic_json(REPORT, report)
                evidence = result['evidence']
                if not any('281,724' in r['text'] or '281.7' in r['text'] for r in evidence):
                    raise ServiceError('Revenue retrieval check failed: expected FY2025 figure absent from selected evidence.')
                text = result['answer']
                if result['status'] != 'grounding_check_passed':
                    raise ServiceError('Answer verification failed: '+(result['trace'][-1]['detail'] if result.get('trace') else result['status']))
                if not re.search(r'\[S[1-5]\]', text) or not any(x in text for x in ('281,724','281.724','281.7')):
                    raise ServiceError('Cited-answer check failed: expected revenue figure and source reference.')
                return result
            stage(f'End-to-end answer · {strategy} · rerank {"on" if rerank else "off"}', research)
    loaded={d['year'] for d in documents}
    if {'2022','2023','2024','2025'}.issubset(loaded):
        def multiyear():
            query='Compare Microsoft total revenue from 2022 to 2025. Show one row per fiscal year, exact reported revenue in USD millions, and a source citation for each row.'
            result=workflow.research(query,'fixed','All years',True,progress)
            report['research'].append(result)
            engine.atomic_json(REPORT,report)
            if result['status']!='grounding_check_passed':
                reason=result['trace'][-1]['detail'] if result.get('trace') else result['status']
                raise ServiceError('Multi-year verification failed: '+reason)
            expected={'2022':'198270','2023':'211915','2024':'245122','2025':'281724'}
            answer=result['answer'].replace(',','')
            for year,amount in expected.items():
                matching=[r for r in result['evidence'] if r['year']==year]
                if not matching or not any(amount in r['text'].replace(',','') for r in matching):
                    raise ServiceError(f'Multi-year check: missing revenue evidence from the FY{year} report.')
                if year not in answer or amount not in answer:
                    raise ServiceError(f'Multi-year check: answer must contain FY{year} and its exact revenue in millions.')
            return result
        stage('Four-year revenue comparison · FY2022–FY2025',multiyear)
    report['status'] = 'passed'
    report['note'] = 'Revenue sanity checks across four configurations, plus a four-year comparison when that corpus is loaded. Automatic checks do not replace human faithfulness evaluation.'
    engine.atomic_json(REPORT, report)
    progress('All demo checks passed. Open Research desk to ask questions.')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    try:
        run(lambda message: print(message, flush=True), rebuild=args.rebuild)
    except ServiceError as error:
        print(str(error))
        raise SystemExit(1)
