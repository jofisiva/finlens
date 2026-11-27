import json
from unittest.mock import patch
import workflow

EVIDENCE=[{'id':'a','title':'Microsoft FY2025','blocks':'1','text':'FY2025 revenue was $281,724 million.','year':'2025'}]
SUFFICIENT={'sufficient':True,'source_ids':[1],'reason':'Revenue and year are present.'}
ANSWER='Revenue was $281,724 million in FY2025 [S1].'
GOOD={'grounded':True,'answers_question':True,'reason':'Supported by cited passage.',
      'claims':[{'claim':'FY2025 revenue was $281,724 million.','supported':True,'source_ids':[1],
                 'evidence_ids':['S1:E1']}]}


def test_verified_answer_and_trace():
    with patch('engine.retrieve',return_value=EVIDENCE),patch('workflow.completion',side_effect=[json.dumps(SUFFICIENT),ANSWER,json.dumps(GOOD)]):
        result=workflow.research('Revenue?')
    assert result['status']=='grounding_check_passed'
    assert result['answer']==ANSWER
    assert len(result['claims'])==1
    assert [t['stage'] for t in result['trace']]==['Retrieval','Evidence sufficiency','Generation','Grounding']


def test_insufficient_evidence_retries_twice_and_never_generates():
    no=json.dumps({'sufficient':False,'source_ids':[],'reason':'Wrong period.'})
    with patch('engine.retrieve',return_value=EVIDENCE) as retrieve,patch('workflow.completion',return_value=no) as llm:
        result=workflow.research('Revenue?',year='2024')
    assert result['status']=='insufficient_evidence'
    assert retrieve.call_count==llm.call_count==2
    assert [c.kwargs['limit'] for c in retrieve.call_args_list]==[5,10]
    assert all(c.args[2]=='2024' for c in retrieve.call_args_list)
    assert not result['claims']


def test_failed_grounding_never_forced_to_pass():
    bad={**GOOD,'grounded':False,'reason':'Wrong units.'}
    with patch('engine.retrieve',return_value=EVIDENCE),patch('workflow.completion',side_effect=[json.dumps(SUFFICIENT),ANSWER,json.dumps(bad),ANSWER,json.dumps(bad)]) as llm:
        result=workflow.research('Revenue?')
    assert llm.call_count==5
    assert result['status']=='verification_failed'
    assert result['answer']==workflow.ABSTAIN
    assert ANSWER not in json.dumps(result)


def test_malformed_grader_fails_closed():
    with patch('engine.retrieve',return_value=EVIDENCE),patch('workflow.completion',return_value='yes') as llm:
        result=workflow.research('Revenue?')
    assert result['status']=='verification_failed'
    assert llm.call_count==1


def test_string_boolean_not_accepted():
    with patch('workflow.object_response',return_value={**SUFFICIENT,'sufficient':'true'}):
        assert workflow.evidence_review('Revenue?',EVIDENCE) is None


def test_checker_fabricated_quote_is_rejected():
    bad={**GOOD,'claims':[{'claim':'Revenue','supported':True,'source_ids':[1],'evidence_ids':['S1:E999']}]}
    with patch('workflow.object_response',return_value=bad):
        assert workflow.grounding_review('Revenue?',ANSWER,EVIDENCE)['passed'] is False


def test_invalid_source_selection_is_rejected():
    with patch('workflow.object_response',return_value={**SUFFICIENT,'source_ids':[9]}):
        assert workflow.evidence_review('Revenue?',EVIDENCE) is None


def test_bad_citation_can_retry_and_succeed():
    with patch('engine.retrieve',return_value=EVIDENCE),patch('workflow.completion',side_effect=[json.dumps(SUFFICIENT),'Revenue [S9]',ANSWER,json.dumps(GOOD)]):
        result=workflow.research('Revenue?')
    assert result['status']=='grounding_check_passed'
    assert len([r for r in result['trace'] if r['stage']=='Generation'])==2


def test_invalid_review_is_not_counted_as_successful_abstention(tmp_path):
    import evaluate
    (tmp_path/'questions.json').write_text(json.dumps([{'question':'Future stock price?','year':'2025','answerable':False}]))
    (tmp_path/'reports').mkdir()
    with patch('engine.ROOT',tmp_path),patch('workflow.research',return_value={'question':'Future stock price?','status':'verification_failed','seconds':1}):
        report=evaluate.run_answers()
    assert report['summary']['Expected outcome matches']==0
    assert report['summary']['Verification failures']==1
    assert report['runs'][0]['human_faithfulness'] is None


def test_table_excerpt_resolves_original_text_without_model_copying():
    table=[{**EVIDENCE[0],'text':'(In millions) | Year | 2025 | 2024 | Total | $ 281,724 | $ 245,122'}]
    verdict={**GOOD,'claims':[{'claim':'Revenue was $281,724 million in FY2025.','supported':True,
                             'source_ids':[1],'evidence_ids':['S1:E1'],'quotes':[{'text':'untrusted paraphrase'}]}]}
    with patch('workflow.object_response',return_value=verdict):
        result=workflow.grounding_review('Revenue?',ANSWER,table)
    assert result['passed']
    assert result['claims'][0]['quotes']==[{'source_id':1,'text':table[0]['text']}]


def test_excerpt_catalog_covers_tail_and_preserves_whitespace():
    text=' | '.join('cell'+str(i) for i in range(100))+'\nLAST'
    _,catalog=workflow.excerpt_catalog([{**EVIDENCE[0],'text':text}])
    assert any('LAST' in row['text'] for row in catalog.values())
    assert all(row['text'] in text for row in catalog.values())


def test_real_langgraph_topology_has_conditional_retry_edges():
    from langgraph.graph.state import CompiledStateGraph
    graph=workflow.build_graph()
    assert isinstance(graph,CompiledStateGraph)
    edges={(edge.source,edge.target):edge.conditional for edge in graph.get_graph().edges}
    assert edges[('grade_evidence','retrieve')]
    assert edges[('grade_evidence','generate')]
    assert edges[('check_grounding','generate')]
    assert edges[('check_grounding','abstain')]
    assert ('finish','__end__') in edges


def test_graph_retries_do_not_leak_draft_or_cross_request_state():
    with patch('engine.retrieve',return_value=EVIDENCE),patch('workflow.completion',side_effect=[json.dumps(SUFFICIENT),'invalid [S99]',ANSWER,json.dumps(GOOD)]):
        first=workflow.research('Revenue?')
    with patch('engine.retrieve',return_value=[]):
        second=workflow.research('Unanswerable?')
    assert first['framework']=='LangGraph StateGraph'
    assert first['generation_attempts']==2
    assert second['generation_attempts']==0
    assert second['retrieval_attempts']==2
    assert second['status']=='insufficient_evidence'
    assert all(k not in first for k in ('draft','feedback','started'))


def test_multiyear_retrieves_each_report_separately():
    state={'question':'Compare revenue from 2022 to 2025','strategy':'fixed','period':'All years',
           'rerank':True,'retrieval_attempts':0,'trace':[]}
    with patch('engine.corpus',return_value=[{'year':str(y)} for y in range(2022,2026)]),patch('engine.retrieve',return_value=[]) as retrieve:
        result=workflow.retrieve_node(state)
    assert [c.args[2] for c in retrieve.call_args_list]==['2022','2023','2024','2025']
    assert result['retrieval_attempts']==1


def test_grounding_retries_invalid_schema_once_same_draft():
    with patch('workflow.object_response',side_effect=[{'grounded':'yes'},GOOD]) as reviewer:
        result=workflow.grounding_review('Revenue?',ANSWER,EVIDENCE)
    assert result['passed']
    assert reviewer.call_count==2
    assert reviewer.call_args_list[0].args[1]==reviewer.call_args_list[1].args[1]


def test_grounding_invalid_schema_has_hard_retry_limit():
    with patch('workflow.object_response',return_value={'grounded':'yes'}) as reviewer:
        assert workflow.grounding_review('Revenue?',ANSWER,EVIDENCE) is None
    assert reviewer.call_count==2


def test_grounding_only_receives_cited_sources():
    with patch('workflow.object_response',return_value=GOOD) as reviewer:
        workflow.grounding_review('Revenue?',ANSWER,EVIDENCE+[dict(EVIDENCE[0])])
    assert [s['source_id'] for s in reviewer.call_args.args[1]['sources']]==[1]


def test_multiyear_selection_preserves_earlier_sources():
    evidence=[{**EVIDENCE[0],'year':str(y),'id':str(y)} for y in range(2022,2026)]
    state={'question':'Compare revenue from 2022 to 2025','evidence':evidence,
           'retrieval_attempts':1,'trace':[]}
    with patch('workflow.evidence_review',return_value=SUFFICIENT):
        update=workflow.grade_evidence_node(state)
    assert 'evidence' not in update  # LangGraph retains all retrieved periods.
    assert update['evidence_sufficient']


def test_missing_years_cannot_pass_even_with_positive_model_review():
    state={'question':'Compare revenue from 2022 to 2025',
           'draft':'FY2024: $245,122 million [S1]. FY2025: $281,724 million [S1].',
           'generation_attempts':1,'trace':[],'evidence':EVIDENCE}
    with patch('workflow.grounding_review') as reviewer:
        update=workflow.check_grounding_node(state)
    reviewer.assert_not_called()
    assert '2022, 2023' in update['feedback']
    assert update['trace'][-1]['status']=='failed'
    state['generation_attempts']=2
    assert workflow.check_grounding_node(state)['status']=='verification_failed'


def test_required_years_include_fy_prefix_and_range():
    assert workflow.requested_years('Compare from FY2022 to FY2025')==['2022','2023','2024','2025']
    assert workflow.requested_years('Compare FY2022 and FY2025')==['2022','2025']
