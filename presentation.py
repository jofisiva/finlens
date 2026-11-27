"""Read-only presentation helpers; financial values and evidence remain unchanged."""
import re


def source_blocks(item, documents):
    ids={int(value) for value in item.get('blocks','').split(',') if value.strip().isdigit()}
    for doc in documents:
        if doc['url']==item.get('url') and doc['year']==item.get('year'):
            return [b for b in doc['blocks'] if b['block'] in ids]
    return []


def table_rows(text):
    rows=[[cell.strip() for cell in line.split('|')] for line in text.splitlines() if line.strip()]
    if not rows:
        return []
    width=max(map(len,rows))
    padded=[row+['']*(width-len(row)) for row in rows]
    # Remove only columns that are empty in every row. Never infer financial headers.
    columns=[i for i in range(width) if any(row[i] for row in padded)]
    return [[row[i] for i in columns] for row in padded]


def readable_record(result):
    lines=['# Financial research brief','',result['question'],'',result['answer'],'',
           f"Status: {result.get('status','unknown')}",f"Response time: {result['seconds']} seconds",'',
           '## Sources','']
    for i,item in enumerate(result['evidence'],1):
        lines.extend([f"### [S{i}] {item['title']}",item['url'],''])
        quotes=[q['text'] for c in result.get('claims',[]) for q in c.get('quotes',[]) if q['source_id']==i]
        for quote in dict.fromkeys(quotes):
            lines.extend(['> '+quote.replace('\n','\n> '),''])
    lines.extend(['## Verification',''])
    for row in result.get('trace',[]):
        lines.append(f"- {row['stage']} (attempt {row['attempt']}): {row['status']}. {row['detail']}")
    lines.extend(['','Automated reviews can miss errors. Verify the source figures, fiscal periods and units.'])
    return '\n'.join(lines)


def relevant_excerpts(text, question, count=3):
    """Rank verbatim source sentences for display; never synthesize an answer."""
    stop={'what','were','was','the','of','in','and','a','an','to','how','did','main'}
    words=set(re.findall(r'[a-z0-9]+',question.lower()))-stop
    if words & {'drivers','growth','increase','increased'}:
        words.update({'driven','increased','growth'})
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])',text) if s.strip()]
    scored=[]
    for index,sentence in enumerate(sentences):
        terms=set(re.findall(r'[a-z0-9]+',sentence.lower()))
        score=len(terms & words)/(1+len(sentence)/800)
        scored.append((score,index,sentence))
    selected=sorted(sorted(scored,key=lambda row:row[0],reverse=True)[:count],key=lambda row:row[1])
    return [row[2] for row in selected] or [text]
