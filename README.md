# FinLens

Financial document research over Microsoft FY2022–FY2025 annual reports. A Streamlit interface presents cited briefings, inspectable source passages and readable exports. LangGraph orchestrates retrieval, evidence review, generation, grounding checks and bounded retries.

## Run locally

Use Python 3.12 and run from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env locally with your provider credentials.
python -m streamlit run app.py
```

Set `PINECONE_API_KEY` and `LLM_API_KEY` (your Nebius API key). The example supplies the regional Nebius endpoint and Qwen model routing key. Credentials are not included in this repository. Never commit `.env` or Streamlit secrets.

In the app, enable the demo and technical tools and run Connection checks. The app downloads source reports, creates the dedicated Pinecone index if needed, and builds both chunking strategies. Provider calls use your account credits. Generated data and local reports are excluded from Git; each new checkout must build its own indexes and local artifacts.

## Pipeline

- Corpus: four English annual reports from Microsoft Investor Relations, FY2022–FY2025. HTML ingestion preserves paragraph/table blocks and source references.
- Fixed chunking: 300 words, 40-word overlap.
- Semantic chunking: embedding-based adjacent paragraph boundaries, 0.75 similarity threshold, 300-word cap.
- Embeddings: Pinecone multilingual-e5-large, 1024 dimensions.
- Hybrid retrieval: dense top 20 plus BM25 top 20, reciprocal rank fusion; optional bge-reranker-v2-m3 reranking.
- Generation: Nebius Qwen/Qwen3-30B-A3B-Instruct-2507.
- Orchestration: actual LangGraph StateGraph, with independent bounded retrieval and generation retries, structured reviews and source excerpt validation.
- Multi-year queries retrieve across requested report years and check that requested years appear in the answer.

Source reports: [2022](https://www.microsoft.com/investor/reports/ar22/), [2023](https://www.microsoft.com/investor/reports/ar23/), [2024](https://www.microsoft.com/investor/reports/ar24/), [2025](https://www.microsoft.com/investor/reports/ar25/).

## Evaluation

```bash
python -m pytest tests -q
python evaluate.py
```

The current question set has 19 cases. Retrieval evaluation runs four configurations (fixed/semantic × reranking off/on), producing 76 runs. Full-answer evaluation is available separately in the app. Numeric-term coverage is not relevance or faithfulness; automatic outcome matching is not human-verified factual accuracy.

Recorded validation:

- 40 local regression tests passed.
- All 10 live setup checks passed, including the four-year revenue comparison.
- Full-answer evaluation: 18/19 expected outcomes matched, zero verification failures, 27.18-second p95 latency. The cash-flow comparison returned insufficient evidence.
- Human faithfulness grading remains pending. The saved earlier retrieval baseline covers 15 questions; the expanded 76-run comparison remains pending.

## Documentation

Project documentation and the recording are supplied separately through the submission links. Local submission documents, recording notes and the Mac launcher are excluded from this repository. Model instructions and workflow behavior are available in `engine.py` and `workflow.py`.

## Limitations

Automated reviews can share the generator's errors. Long tables may lose header context when split. This prototype has no arbitrary PDF upload, investment recommendation service, authentication or multi-user deployment. The original 15-second p95 target has not been achieved in the full-answer evaluation.
