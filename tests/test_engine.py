import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine


def test_fixed_chunks_bounded_and_cover_tail():
    doc = {'id':'test', 'year':'2025', 'title':'Test', 'url':'https://example.com',
           'blocks':[{'text':' '.join(f'word{i}' for i in range(701)), 'block':0, 'kind':'paragraph'}]}
    rows = engine.chunks([doc])
    assert all(len(r['text'].split()) <= 300 for r in rows)
    assert 'word700' in rows[-1]['text']
    assert len(set(r['id'] for r in rows)) == len(rows)


def test_year_filter_and_revenue_evidence():
    hits = engine.retrieve('What was total revenue in fiscal 2025?', year='2025')
    assert hits and all(h['year'] == '2025' for h in hits)
    assert any('281,724' in h['text'] for h in hits)


def test_empty_evidence_abstains():
    assert 'could not find' in engine.answer('Predict stock price', [])


def test_missing_provider_never_simulates_generation():
    with patch.dict(engine.os.environ, {'LLM_API_KEY':'', 'LLM_MODEL':''}):
        assert 'Evidence preview only' in engine.answer('Revenue?', [{'text':'sample'}])


def test_semantic_boundaries_use_embeddings():
    doc = {'id':'test', 'year':'2025', 'title':'Test', 'url':'https://example.com',
           'blocks':[{'text':'Revenue increased.', 'block':0, 'kind':'paragraph'},
                     {'text':'Cybersecurity threats.', 'block':1, 'kind':'paragraph'}]}
    with patch('engine.embed', return_value=[[1,0],[0,1]]):
        assert len(engine.chunks([doc], 'semantic', live=True)) == 2
