import html
import json
import os
import time
from pathlib import Path
import streamlit as st
import engine
from presentation import source_blocks, table_rows, readable_record, relevant_excerpts
from services import ServiceError

st.set_page_config(page_title='FinLens | Leadership briefing', page_icon='◈', layout='wide')
st.markdown('''<style>
.stApp{background:#0b1220;color:#e7eef8}h1,h2,h3{color:#f2f6ff}
[data-testid="stSidebar"]{background:#111d30}.stButton>button{border-radius:10px;min-height:44px}button[kind="primary"]{color:#09231d !important;font-weight:600}
.hero h1{font-size:clamp(1.8rem,3vw,2.7rem);line-height:1.2}.hero{padding:4px 0 20px;border-bottom:1px solid #27364b;margin-bottom:25px}
.eyebrow{color:#57d3b0;letter-spacing:3px;font-size:14px}.hero p{color:#aebdd0;font-size:19px;max-width:850px}
.evidence{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.75;font-size:16px;color:#d8e4f3}
[data-testid="stMetric"]{background:#142238;border:1px solid #24364f;border-radius:12px;padding:16px}
</style>''', unsafe_allow_html=True)


def read_report(name):
    path = engine.ROOT / 'reports' / name
    return json.loads(path.read_text()) if path.exists() else None


def fail(error):
    st.error(str(error) if isinstance(error, ServiceError) else f'{type(error).__name__}: operation failed. Check configuration and retry.')


try:
    manifest = engine.load_manifest()
except (ServiceError, ValueError):
    manifest = None

loaded_years=sorted({d['year'] for d in engine.corpus()},reverse=True)
coverage='–'.join([f'FY{min(loaded_years)}',f'FY{max(loaded_years)}']) if loaded_years else 'No reports loaded'

with st.sidebar:
    st.title('◈ FinLens')
    st.caption('FINANCIAL RESEARCH FOR LEADERS')
    technical = st.checkbox('Show demo & technical tools',value=False)
    pages=['Research desk','Source library']
    if technical:
        pages=['Research desk','Demo setup','Source library','Evaluation lab','Project guide']
    labels={'Research desk':'Executive briefing','Source library':'Source reports',
            'Demo setup':'Connection checks','Evaluation lab':'Quality evaluation','Project guide':'Architecture & demo'}
    page=st.radio('Workspace',pages,format_func=lambda value:labels[value])
    st.divider()
    st.write('**Company**')
    st.write('Microsoft')
    year=st.selectbox('Reporting period',['All years']+loaded_years)
    live=bool(manifest)
    strategy='fixed'
    rerank=bool(manifest)
    if technical:
        with st.expander('Retrieval settings',expanded=False):
            live=st.toggle('Pinecone hybrid retrieval',value=bool(manifest),disabled=not bool(manifest))
            strategy=st.selectbox('Chunk strategy',['fixed','semantic'],disabled=not live)
            rerank=st.toggle('Neural reranking',value=bool(manifest),disabled=not live)
        st.caption('Orchestrated with LangGraph')
        if manifest:
            st.caption(f"{sum(manifest['counts'].values()):,} indexed chunks · two strategies")
    st.divider()
    st.caption(f'Loaded coverage: {coverage}. Historical disclosures; not live market data.')
    if not manifest:
        st.info('Setup is needed. Enable demo tools and open Connection checks.')

if page == 'Research desk':
    st.markdown('<div class="hero"><div class="eyebrow">MICROSOFT · EXECUTIVE RESEARCH</div><h1>Prepare for the next financial discussion.</h1><p>Explore performance, growth drivers and disclosed risks—with the source evidence beside each answer.</p></div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric('Company','Microsoft')
    b.metric('Reporting coverage',coverage)
    c.metric('Source reports',len(engine.corpus()))
    st.write('')
    examples = ['Write your own question','What was Microsoft’s total revenue in fiscal 2025?',
                'How did Microsoft Cloud revenue change in fiscal 2025?',
                'What were the main drivers of revenue growth in fiscal 2025?',
                'What cybersecurity risks does Microsoft disclose?']
    if {'2022','2023','2024','2025'}.issubset(set(loaded_years)):
        examples.insert(1,'Compare Microsoft total revenue from 2022 to 2025 in a table, with a citation for each year.')
    selected = st.selectbox('Start with a business question',examples)
    with st.form('research'):
        query = st.text_input('What would you like to understand?',value='' if selected==examples[0] else selected,
                              placeholder='Ask about revenue, cash flow, or disclosed risks…')
        submitted = st.form_submit_button('Prepare briefing',type='primary')
    if submitted:
        st.session_state.pop('result',None)
        if not query.strip():
            st.warning('Enter a research question first.')
        elif len(query)>2000:
            st.warning('Keep the question under 2,000 characters.')
        else:
            with st.spinner('Finding evidence and preparing the answer…'):
                try:
                    start=time.monotonic()
                    if live:
                        from workflow import research
                        progress=st.empty()
                        result=research(query,strategy,year,rerank,progress.info if technical else lambda _:None)
                        progress.empty()
                    else:
                        evidence=engine.retrieve(query,live=False,year=year)
                        result={'question':query,'answer':'Evidence preview only. Enable Pinecone hybrid retrieval after Demo setup for a generated, cited answer.',
                                'evidence':evidence,'seconds':round(time.monotonic()-start,2),
                                'mode':'Local BM25 preview','strategy':'fixed','rerank':False,'period':year,'status':'preview'}
                    engine.atomic_json(engine.ROOT/'reports'/'latest-research.json',result)
                    st.session_state['result']=result
                    st.session_state.setdefault('history',[]).insert(0,result)
                    st.session_state['history']=st.session_state['history'][:10]
                except Exception as error:
                    fail(error)
    if result:=st.session_state.get('result'):
        st.divider()
        from workflow import VERSION
        if result.get('workflow_version') and result['workflow_version'] != VERSION:
            st.info('This brief used an earlier workflow. Prepare it again to apply the latest checks.')
        left,right=st.columns([1.1,1])
        with left:
            st.subheader('Executive briefing')
            st.write(result['question'])
            if result.get('status')=='grounding_check_passed':
                st.success('Automated evidence checks passed')
            elif result.get('status')=='verification_failed':
                st.warning('Answer verification failed · source passages remain available')
            elif result.get('status')=='insufficient_evidence':
                st.warning('Retrieved evidence does not fully answer this question')
            with st.container(border=True):
                st.markdown(result['answer'].replace('$', r'\$'))
            st.caption(f"Prepared in {result['seconds']} seconds · Based on the selected annual reports")
            if technical:
                st.caption(f"{result['mode']} · {result['strategy']} chunks · rerank {'on' if result['rerank'] else 'off'}")
            st.download_button('Download readable brief',readable_record(result),'finlens-research.md','text/markdown')
            if technical:
                with st.expander('Export technical record'):
                    st.download_button('Download JSON',json.dumps(result,indent=2),'finlens-research.json','application/json')
            st.caption('Automated model reviews can miss errors. Inspect source passages to verify meaning, fiscal period and units.')
            if result.get('trace') and (technical or result.get('status') in ('verification_failed','insufficient_evidence')):
                with st.expander('How this answer was checked',expanded=False):
                    for step in result['trace']:
                        st.markdown(f"**{step['stage']} · {step['status']}** (attempt {step['attempt']})")
                        st.write(step['detail'])
            if result.get('claims'):
                with st.expander('Claims and supporting evidence'):
                    st.caption('Claims are grouped by the source excerpts selected by the automated reviewer. These associations still require human review.')
                    groups={}
                    for claim in result['claims']:
                        key=tuple(sorted(set((q['source_id'],q['text']) for q in claim.get('quotes',[]))))
                        groups.setdefault(key,[]).append(claim['claim'])
                    for quotes,claims in groups.items():
                        with st.container(border=True):
                            st.markdown('**Claims reviewed**')
                            for claim_text in dict.fromkeys(claims):
                                st.markdown('<div class="evidence">• '+html.escape(claim_text)+'</div>',unsafe_allow_html=True)
                            st.markdown('**Selected source evidence**')
                            for source_id,quote_text in quotes:
                                st.caption(f'Source [S{source_id}]')
                                st.markdown('<div class="evidence">'+html.escape(quote_text)+'</div>',unsafe_allow_html=True)
        with right:
            st.subheader('Supporting sources')
            for i,item in enumerate(result['evidence'],1):
                with st.expander(f"[S{i}] {item['title']}",expanded=i==1):
                    excerpts,tables,raw=st.tabs(['Supporting excerpts','Original tables','Raw passage'])
                    with excerpts:
                        quotes=list(dict.fromkeys(q['text'] for claim in result.get('claims',[]) for q in claim.get('quotes',[]) if q['source_id']==i))
                        if not quotes:
                            quotes=relevant_excerpts(item['text'],result['question'])
                            st.caption('Relevant source sentences, shown verbatim—not a verified answer. Open Raw passage for the complete context.')
                        for quote in quotes:
                            st.markdown('<div class="evidence">'+html.escape(quote)+'</div>',unsafe_allow_html=True)
                            st.write('')
                    with tables:
                        originals=[block for block in source_blocks(item,engine.corpus()) if block['kind']=='table']
                        if not originals:
                            st.caption('No original table is associated with this passage.')
                        for block in originals:
                            st.caption(f"Original source table · block {block['block']}. Header rows and units are retained as source data.")
                            rows=table_rows(block['text'])
                            if rows:
                                st.dataframe(rows,use_container_width=True,hide_index=True)
                    with raw:
                        st.markdown('<div class="evidence">'+html.escape(item['text'])+'</div>',unsafe_allow_html=True)
                        st.caption(f"Source blocks {item['blocks']} · chunk {item['id']}")
                        if 'rerank_score' in item:
                            st.caption(f"Reranker score: {item['rerank_score']:.3f} (not a probability of correctness)")
                    st.link_button('Open publisher report',item['url'])
    history=st.session_state.get('history',[])
    if len(history)>1:
        with st.expander('Session research history'):
            for i,entry in enumerate(history):
                if st.button(entry['question'],key=f'history-{i}'):
                    st.session_state['result']=entry
                    st.rerun()

elif page == 'Demo setup':
    st.markdown('<div class="hero"><div class="eyebrow">VERIFY BEFORE YOU DEMO</div><h1>One workspace. Every step checked.</h1><p>Test the services, index the reports, and verify a cited answer through all four retrieval configurations.</p></div>',unsafe_allow_html=True)
    st.write('Uses your configured Pinecone and Nebius accounts. The first run embeds both reports and can take several minutes. API calls use your provider credits. Existing unrelated indexes and namespaces are not deleted.')
    a,b,c=st.columns(3)
    a.metric('Pinecone key','Configured' if os.getenv('PINECONE_API_KEY') else 'Missing')
    b.metric('Nebius key','Configured' if os.getenv('LLM_API_KEY') else 'Missing')
    c.metric('Corpus index','Ready' if manifest else 'Setup needed')
    rebuild=st.checkbox('Rebuild both chunking strategies',value=False)
    if st.button('Run end-to-end setup & checks',type='primary'):
        import demo
        with st.status('Running live checks…',expanded=True) as status:
            progress=st.empty()
            try:
                report=demo.run(progress.info,rebuild)
                status.update(label='End-to-end checks passed',state='complete')
                st.success('Open Research desk. Hybrid retrieval will be enabled when the page refreshes.')
            except Exception as error:
                status.update(label='A check needs attention',state='error')
                fail(error)
    if report:=read_report('demo-checks.json'):
        st.subheader('Latest live check results')
        st.caption(f"Started {report['started_at']} · status: {report['status']}")
        from workflow import VERSION
        if report.get('workflow_version') != VERSION:
            st.warning('These checks predate the current verification workflow. Run setup again; rebuilding the index is not required.')
        st.dataframe(report['checks'],use_container_width=True,hide_index=True)
        if report.get('corpus_digest') and report['corpus_digest']!=engine.corpus_digest(engine.corpus()):
            st.warning('These checks used an earlier corpus. Run setup again.')
        if report.get('research'):
            st.subheader('Research results and verification details')
            for row in report['research']:
                with st.expander(f"{row['strategy']} · rerank {row['rerank']} · {row['seconds']}s"):
                    st.caption(row.get('status','unknown'))
                    st.markdown(row['answer'])
                    if row.get('trace'):
                        st.dataframe(row['trace'],use_container_width=True,hide_index=True)
                    for source_index,source in enumerate(row.get('evidence',[]),1):
                        st.caption(f"[S{source_index}] {source['title']}")
                        st.text(source['text'])
        st.download_button('Download verification report',json.dumps(report,indent=2),'finlens-demo-checks.json','application/json')
    else:
        st.info('No live checks have been recorded yet. A configured key alone does not verify the retrieval pipeline.')

elif page == 'Source library':
    st.title('Microsoft annual reports')
    st.write(f"Browse the {len(engine.corpus())} loaded annual reports by fiscal year. Open the Microsoft webpage or download the original report below. Microsoft provides these downloads in Word format.")
    for doc in engine.corpus():
        with st.container(border=True):
            a,b=st.columns([3,1])
            with a:
                st.subheader(doc['title'])
                st.caption(f"{len(doc['blocks'])} source blocks · {sum(r['kind']=='table' for r in doc['blocks'])} tables")
            with b:
                st.link_button('Read on Microsoft.com',doc['url'])
            original=engine.DATA/'original_reports'/f"Microsoft-FY{doc['year']}-Annual-Report.docx"
            if original.exists():
                st.download_button(f"Download FY{doc['year']} annual report (Word)",original.read_bytes(),original.name,
                                   'application/vnd.openxmlformats-officedocument.wordprocessingml.document',key=f"original-{doc['year']}")
            with st.expander('Inspect source blocks'):
                block_id=st.number_input('Block number',0,len(doc['blocks'])-1,0,key=doc['id'])
                block=doc['blocks'][block_id]
                st.caption(block['kind'])
                st.text(block['text'])
    st.subheader('Add Microsoft reporting years')
    requested_years=st.multiselect('Annual reports to download',engine.SUPPORTED_YEARS,
        default=[y for y in engine.SUPPORTED_YEARS if str(y) not in loaded_years])
    st.caption('Downloads are merged with existing reports. Select an already-loaded year to refresh it. After changing the library, run Connection checks to rebuild the corpus index.')
    if st.button('Download selected annual reports',disabled=not requested_years):
        with st.spinner('Downloading and cleaning reports…'):
            try:
                engine.acquire(requested_years)
                st.rerun()
            except Exception as error:
                fail(error)
    st.caption('If report content changes, Demo setup must rebuild the index before hybrid research can resume.')

elif page == 'Evaluation lab':
    st.title('Measure what retrieval changes.')
    evaluation_cases=json.loads((engine.ROOT/'questions.json').read_text())
    case_count=len(evaluation_cases)
    numeric_count=sum(bool(q['expected_terms']) for q in evaluation_cases)
    st.write(f'Same {case_count} questions. Two chunking strategies. Reranking on and off. Compare measured numeric-term coverage and retrieval latency; review the passages before drawing conclusions.')
    st.info(f'Numeric-term coverage measures whether expected strings occur in the top five passages. It is not relevance, recall, or answer faithfulness. {numeric_count} numeric questions are scored; the others require human review.')
    if st.button(f'Run {case_count*4}-query retrieval comparison',disabled=not bool(manifest),type='primary'):
        import evaluate
        progress=st.empty()
        try:
            with st.spinner('Running the four configurations…'):
                evaluate.run(progress.info)
            st.success('Retrieval comparison complete. Review results below.')
        except Exception as error:
            fail(error)
    if report:=read_report('evaluation.json'):
        st.caption(report['status'])
        if report.get('summary'):
            st.dataframe(report['summary'],use_container_width=True,hide_index=True)
            st.bar_chart([{'Configuration':f"{r['Chunking']} / rerank {r['Reranking']}", 'Coverage':r['Numeric term coverage @5']} for r in report['summary']], x='Configuration', y='Coverage')
        runs=report.get('runs',[])
        if runs:
            chosen=st.selectbox('Inspect an evaluated question',list(dict.fromkeys(r['question'] for r in runs)))
            for row in [r for r in runs if r['question']==chosen]:
                with st.expander(f"{row['strategy']} · rerank {row['rerank']} · {row['seconds']}s"):
                    for hit in row['hits']:
                        st.write(hit['title'])
                        st.text(hit['text'])
        st.download_button('Download comparison report',json.dumps(report,indent=2),'finlens-evaluation.json','application/json')
    else:
        st.warning('No evaluation results recorded. The 95% faithfulness target has not been measured.')
    st.divider()
    st.subheader('Full-answer and abstention evaluation')
    st.write(f'Run all {case_count} questions through evidence review, generation and grounding checks. Uses the sidebar chunking and reranking settings. This makes multiple model calls per question and uses provider credits.')
    if st.button('Evaluate verified answers and abstentions',disabled=not bool(manifest)):
        import evaluate
        progress=st.empty()
        try:
            evaluate.run_answers(progress.info,strategy,rerank)
            st.success('Answer evaluation complete; human claim grading remains pending.')
        except Exception as error:
            fail(error)
    if answers:=read_report('answer-evaluation.json'):
        st.caption(answers['status'])
        if answers.get('summary'):
            st.json(answers['summary'])
        st.dataframe([{'Question':r['question'],'Result':r['status'],
                       'Expected outcome matched':r['expected_outcome_matched'],'Seconds':r['seconds']}
                      for r in answers['runs']],use_container_width=True,hide_index=True)
        st.caption('Expected-outcome matches measure answer/abstain behavior, not factual correctness. Grounding checks are model judgments, not human verification.')
        st.download_button('Download answer evaluation',json.dumps(answers,indent=2),'finlens-answer-evaluation.json','application/json')


else:
    st.title('A research assistant you can explain.')
    st.write('FinLens helps financial analysts answer historical performance and disclosed-risk questions from Microsoft annual reports selected in the source library. Target: 95% human-reviewed claim faithfulness and p95 answer latency below 15 seconds. Targets are not measured results.')
    st.subheader('LangGraph workflow')
    st.write('A compiled StateGraph carries the question, evidence, draft and review results between nodes. Conditional edges retry retrieval or generation at most once, or route to abstention. State is request-scoped; there is no persistent conversation checkpointer.')
    from workflow import build_graph
    topology=build_graph().get_graph()
    lines=['digraph FinLens {', 'rankdir=TB;', 'node [shape=box, style=rounded];']
    for edge in topology.edges:
        style='dashed' if edge.conditional else 'solid'
        lines.append(f'{json.dumps(edge.source)} -> {json.dumps(edge.target)} [style={style}];')
    lines.append('}')
    st.graphviz_chart('\n'.join(lines))
    st.caption('Generated from the compiled graph. Dashed edges are conditional transitions.')
    st.subheader('How the answer is built')
    st.code('Reports → clean paragraphs and tables → fixed / semantic chunks\n→ Pinecone E5 embeddings → vector search + BM25 → rank fusion\n→ optional neural reranking → evidence sufficiency check\n→ Nebius answer → claim grounding review → answer or abstain')
    st.subheader('Five-minute demo plan')
    for title,description in [('0:00–0:35 · The problem','Explain the cost of searching long filings and verifying financial claims.'),
        ('0:35–1:45 · Research desk','Ask the revenue question. Inspect a citation and open the original annual report.'),
        ('1:45–2:45 · Retrieval experiment','Show actual fixed/semantic and reranking results. Explain improvements and regressions.'),
        ('2:45–3:25 · Evidence limits','Ask about a future share price. Show the response and explain the corpus boundary.'),
        ('3:25–4:50 · Implementation and learnings','Explain Pinecone, Nebius, AI-assisted coding, tests, and remaining limitations.')]:
        with st.container(border=True):
            st.write('**'+title+'**')
            st.write(description)
    st.caption('The app is a local single-user prototype. Final submission also needs a reviewed comparison report, project documentation, code assets, and a recorded demo.')
