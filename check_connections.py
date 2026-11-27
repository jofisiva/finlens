"""Check configured services without printing credentials or response bodies."""
from pathlib import Path
import sys
import requests
from dotenv import dotenv_values


def main():
    config = dotenv_values(Path(__file__).resolve().parent / '.env')
    required = ['PINECONE_API_KEY', 'LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL']
    missing = [key for key in required if not config.get(key)]
    if missing:
        print('Missing settings: ' + ', '.join(missing))
        return 1
    base = config['LLM_BASE_URL'].rstrip('/')
    if not base.startswith('https://'):
        print('LLM_BASE_URL must use HTTPS.')
        return 1
    passed = True
    for label, url, headers in [
        ('Pinecone', 'https://api.pinecone.io/indexes',
         {'Api-Key': config['PINECONE_API_KEY'], 'X-Pinecone-API-Version': '2025-04'}),
        ('Nebius', base + '/models',
         {'Authorization': 'Bearer ' + config['LLM_API_KEY']}),
    ]:
        try:
            response = requests.get(url, headers=headers, timeout=25, allow_redirects=False)
            print(f'{label}: HTTP {response.status_code}')
            if not response.ok:
                passed = False
                print('Check the API key/account access for 401 or 403; quota for 429.')
                continue
            if label == 'Nebius':
                available = any(row.get('id') == config['LLM_MODEL']
                                for row in response.json().get('data', []))
                print('Selected model listed: ' + ('YES' if available else 'NO'))
                generation = requests.post(base + '/chat/completions', headers=headers,
                    json={'model': config['LLM_MODEL'],
                          'messages': [{'role': 'user', 'content': 'Reply with exactly OK'}],
                          'max_tokens': 16}, timeout=30, allow_redirects=False)
                print(f'Nebius generation: HTTP {generation.status_code}')
                valid = False
                if generation.ok:
                    choices = generation.json().get('choices', [])
                    valid = bool(choices and choices[0].get('message', {}).get('content'))
                print('Model returned text: ' + ('YES' if valid else 'NO'))
                passed = passed and valid
        except requests.RequestException:
            passed = False
            print(f'{label}: network request failed. Check DNS/network connectivity.')
        except ValueError:
            passed = False
            print(f'{label}: unexpected response format.')
    print('Connection checks passed.' if passed else 'Connection checks incomplete or failed.')
    print('This checks access and a short generation call; embeddings and reranking still require live tests.')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
