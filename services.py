"""Small REST clients. Errors deliberately omit keys and provider response bodies."""
import os
import time
from urllib.parse import urlparse
import requests


class ServiceError(RuntimeError):
    pass


def request(service, method, url, headers, payload=None):
    for attempt in range(3):
        try:
            response = requests.request(method, url, headers=headers, json=payload,
                                        timeout=(10, 60), allow_redirects=False)
        except requests.RequestException:
            raise ServiceError(f'{service}: connection failed. Check network access and retry.') from None
        if response.status_code in (429, 502, 503, 504) and attempt < 2:
            time.sleep(2 ** attempt)
            continue
        if not response.ok:
            hints = {401:'API key was rejected.', 403:'Account lacks permission.',
                     404:'Endpoint, index or model was not found.', 429:'Rate limit or quota reached.',
                     400:'Request or index configuration is incompatible.', 422:'Request validation failed.'}
            raise ServiceError(f'{service}: HTTP {response.status_code}. ' + hints.get(response.status_code, 'Check provider status and configuration.'))
        try:
            return response.json()
        except ValueError:
            raise ServiceError(f'{service}: unexpected response format.') from None


class PineconeREST:
    def __init__(self):
        key = os.getenv('PINECONE_API_KEY')
        if not key:
            raise ServiceError('Pinecone: configure PINECONE_API_KEY in .env.')
        self.headers = {'Api-Key':key, 'X-Pinecone-API-Version':'2025-04', 'Content-Type':'application/json'}

    def control(self, path, payload=None):
        return request('Pinecone', 'GET' if payload is None else 'POST',
                       'https://api.pinecone.io'+path, self.headers, payload)

    def data(self, host, path, payload):
        host = host.removeprefix('https://').rstrip('/')
        if not host.endswith('.pinecone.io') or '/' in host:
            raise ServiceError('Pinecone: invalid index host.')
        return request('Pinecone', 'POST', 'https://'+host+path, self.headers, payload)

    def embed(self, texts, model, kind):
        result = self.control('/embed', {'model':model, 'inputs':[{'text':text} for text in texts],
                                        'parameters':{'input_type':kind, 'truncate':'END'}})
        rows = result.get('data', [])
        vectors = [row['values'] for row in rows]
        if len(vectors) != len(texts) or any(len(v) != 1024 for v in vectors):
            raise ServiceError('Pinecone: embedding response count or dimensions did not match.')
        return vectors

    def rerank(self, query, texts, top_n=5):
        result = self.control('/rerank', {'model':'bge-reranker-v2-m3', 'query':query,
            'documents':[{'id':str(i), 'text':text} for i,text in enumerate(texts)],
            'top_n':min(top_n,len(texts)), 'return_documents':False, 'parameters':{'truncate':'END'}})
        rows = result.get('data', [])
        if not rows or any(not isinstance(r.get('index'), int) or not 0 <= r['index'] < len(texts) for r in rows):
            raise ServiceError('Pinecone: invalid reranking response.')
        return rows


def completion(messages, max_tokens=1200, json_mode=False):
    base = os.getenv('LLM_BASE_URL', '').rstrip('/')
    if urlparse(base).scheme != 'https':
        raise ServiceError('Nebius: configure an HTTPS LLM_BASE_URL.')
    result = request('Nebius', 'POST', base+'/chat/completions',
        {'Authorization':'Bearer '+os.getenv('LLM_API_KEY',''), 'Content-Type':'application/json'},
        {'model':os.getenv('LLM_MODEL',''), 'temperature':0, 'max_tokens':max_tokens, 'messages':messages,
         **({'response_format':{'type':'json_object'}} if json_mode else {})})
    choices = result.get('choices', [])
    if not choices or not choices[0].get('message',{}).get('content'):
        raise ServiceError('Nebius: no answer text returned.')
    if choices[0].get('finish_reason') == 'length':
        raise ServiceError('Nebius: answer was truncated. Narrow the question and retry.')
    return choices[0]['message']['content']
