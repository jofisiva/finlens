import json
import pytest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import engine

APP=str(engine.ROOT/'app.py')

@pytest.fixture(autouse=True)
def prevent_test_report_writes():
    with patch('engine.atomic_json'):
        yield



def test_all_screens_and_offline_question():
    app=AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    app.sidebar.checkbox[0].set_value(True).run(timeout=20)
    app.sidebar.toggle[0].set_value(False).run(timeout=20)
    app.text_input[0].set_value('What was total revenue in fiscal 2025?')
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert any('Evidence preview only' in row.value for row in app.markdown)
    for page in ['Demo setup','Source library','Evaluation lab','Project guide']:
        app.sidebar.radio[0].set_value(page).run(timeout=20)
        assert not app.exception


def test_evaluation_chart_with_saved_results(tmp_path):
    (tmp_path/'questions.json').write_text((engine.ROOT/'questions.json').read_text())
    reports=tmp_path/'reports'
    reports.mkdir()
    (reports/'evaluation.json').write_text(json.dumps({'status':'test fixture','runs':[],
       'summary':[{'Chunking':'fixed','Reranking':'Off','Numeric term coverage @5':0.5,'Retrieval p95 (s)':1,'Queries':15,'Numeric questions':6}]}))
    with patch('engine.ROOT',tmp_path):
        app=AppTest.from_file(APP).run(timeout=20)
        app.sidebar.checkbox[0].set_value(True).run(timeout=20)
        app.sidebar.radio[0].set_value('Evaluation lab').run(timeout=20)
        assert not app.exception
        assert len(app.dataframe)==1


def test_live_research_displays_audit_without_network():
    result={'question':'Revenue?','answer':'Revenue was 100 [S1].','evidence':[],
            'status':'grounding_check_passed','seconds':1.2,'mode':'Pinecone hybrid',
            'strategy':'fixed','rerank':True,'trace':[{'stage':'Grounding','attempt':1,'status':'passed','detail':'Supported'}],
            'claims':[{'claim':'Revenue was 100','quotes':[{'source_id':1,'text':'Revenue 100'}]}]}
    with patch('engine.load_manifest',return_value={'counts':{'fixed':1,'semantic':1}}),patch('workflow.research',return_value=result):
        app=AppTest.from_file(APP).run(timeout=20)
        app.sidebar.checkbox[0].set_value(True).run(timeout=20)
        app.text_input[0].set_value('Revenue?')
        app.button[0].click().run(timeout=20)
        assert not app.exception
        assert any('Automated evidence checks passed' in row.value for row in app.success)
        assert any('Grounding · passed' in row.value for row in app.markdown)
