from unittest.mock import patch, Mock
import pytest
import engine
from services import PineconeREST, ServiceError, completion, request


def test_embedding_contract_and_shape():
    with patch.dict('os.environ', {'PINECONE_API_KEY':'test'}), patch('services.request',return_value={'data':[{'values':[0.0]*1024}]}) as call:
        vectors=PineconeREST().embed(['revenue'],engine.EMBED_MODEL,'passage')
        assert len(vectors[0])==1024
        body=call.call_args.args[-1]
        assert body['inputs']==[{'text':'revenue'}]
        assert body['parameters']['input_type']=='passage'


def test_embedding_mismatch_fails():
    with patch.dict('os.environ', {'PINECONE_API_KEY':'test'}), patch('services.request',return_value={'data':[]}):
        with pytest.raises(ServiceError):
            PineconeREST().embed(['revenue'],engine.EMBED_MODEL,'passage')


def test_rerank_bad_index_fails():
    with patch.dict('os.environ', {'PINECONE_API_KEY':'test'}), patch('services.request',return_value={'data':[{'index':4,'score':1}]}):
        with pytest.raises(ServiceError):
            PineconeREST().rerank('revenue',['revenue'])


def test_provider_error_never_contains_secret_or_body():
    response=Mock(ok=False,status_code=401,text='sensitive provider body')
    with patch('services.requests.request',return_value=response):
        with pytest.raises(ServiceError) as error:
            request('Pinecone','GET','https://api.pinecone.io/indexes',{'Api-Key':'secret'})
        assert 'secret' not in str(error.value)
        assert 'sensitive' not in str(error.value)
        assert '401' in str(error.value)


def test_truncated_generation_fails():
    with patch.dict('os.environ',{'LLM_BASE_URL':'https://example.com/v1'}), patch('services.request',return_value={'choices':[{'message':{'content':'partial'},'finish_reason':'length'}]}):
        with pytest.raises(ServiceError):
            completion([])


def test_wrong_citation_rejected():
    with patch.dict('os.environ',{'LLM_API_KEY':'test','LLM_MODEL':'test'}), patch('engine.completion',return_value='Revenue was 100 [S9].'):
        result=engine.answer('Revenue?',[{'title':'Report','blocks':'1','text':'Revenue was 100.'}])
        assert 'invalid source reference' in result


def test_changed_corpus_invalidates_manifest(tmp_path):
    engine.atomic_json(tmp_path/'manifest.json',{'corpus_digest':'old'})
    with patch('engine.DATA',tmp_path),patch('engine.corpus',return_value=[]):
        with pytest.raises(ServiceError,match='changed'):
            engine.load_manifest()


def test_demo_failure_saved_without_false_success(tmp_path):
    import demo
    with patch('demo.REPORT',tmp_path/'demo.json'), patch('demo.PineconeREST') as client:
        client.return_value.control.side_effect=ServiceError('Pinecone: HTTP 401.')
        with pytest.raises(ServiceError):
            demo.run()
    import json
    report=json.loads((tmp_path/'demo.json').read_text())
    assert report['status']=='failed'
    assert report['checks'][0]['status']=='failed'
    assert not report['research']


def test_incompatible_existing_index_never_upserts():
    with patch.dict('os.environ',{'PINECONE_INDEX':'finlens'}),patch('engine.PineconeREST') as factory:
        client=factory.return_value
        client.control.side_effect=[{'indexes':[{'name':'finlens'}]}, {'dimension':1536,'metric':'cosine','status':{'ready':True}}]
        with pytest.raises(ServiceError,match='1024'):
            engine.index_corpus([{'id':'test'}])
        client.data.assert_not_called()


def test_live_fusion_rerank_preserves_metadata(tmp_path):
    import json
    records=[{'id':'a','text':'revenue 100','title':'Report','year':'2025','blocks':'1','url':'https://example.com'},
             {'id':'b','text':'cloud revenue 200','title':'Report','year':'2025','blocks':'2','url':'https://example.com'}]
    (tmp_path/'fixed.json').write_text(json.dumps(records))
    manifest={'host':'test.pinecone.io','namespaces':{'fixed':'test-fixed'}}
    with patch('engine.DATA',tmp_path),patch('engine.load_manifest',return_value=manifest),patch('engine.embed',return_value=[[0.0]*1024]),patch('engine.PineconeREST') as factory:
        client=factory.return_value
        client.data.return_value={'matches':[{'metadata':records[0]},{'metadata':records[1]}]}
        client.rerank.return_value=[{'index':1,'score':0.9},{'index':0,'score':0.3}]
        result=engine.retrieve('revenue','fixed','2025',True,True)
        assert len(result)==2
        assert result[0]['rerank_score']==0.9
        assert {r['id'] for r in result}=={'a','b'}
        payload=client.data.call_args.args[2]
        assert payload['includeMetadata'] is True
        assert payload['filter']=={'year':{'$eq':'2025'}}


def test_empty_dense_results_do_not_silently_fall_back(tmp_path):
    (tmp_path/'fixed.json').write_text('[]')
    with patch('engine.DATA',tmp_path),patch('engine.load_manifest',return_value={'host':'test.pinecone.io','namespaces':{'fixed':'test'}}),patch('engine.embed',return_value=[[0.0]*1024]),patch('engine.PineconeREST') as factory:
        factory.return_value.data.return_value={'matches':[]}
        with pytest.raises(ServiceError,match='no vectors'):
            engine.retrieve('revenue',live=True)


def test_demo_keeps_failed_research_trace(tmp_path):
    import demo
    import json
    for name in ('fixed','semantic'):
        (tmp_path/f'{name}.json').write_text('[]')
    failed={'evidence':[{'text':'281,724'}],'answer':'Withheld','status':'verification_failed',
            'trace':[{'detail':'Unknown excerpt ID'}]}
    with patch('demo.REPORT',tmp_path/'report.json'),patch('engine.DATA',tmp_path),patch('engine.corpus',return_value=[{'id':'test'}]),patch('engine.load_manifest',return_value={'counts':{'fixed':1,'semantic':1}}),patch('demo.PineconeREST') as pc,patch('engine.embed',return_value=[[0.0]*1024]),patch('demo.completion',return_value='OK'),patch('workflow.research',return_value=failed):
        pc.return_value.rerank.return_value=[{'index':0}]
        with pytest.raises(ServiceError,match='Unknown excerpt ID'):
            demo.run()
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['status']=='failed'
    assert report['research'][0]['trace']==failed['trace']
