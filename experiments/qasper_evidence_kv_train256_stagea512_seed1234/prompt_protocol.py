import math

SYSTEM = ('Answer the question using only the supplied paper. Return only a concise '
          'answer, not an explanation or reasoning. For a yes/no question return exactly '
          'Yes or No. If the paper does not contain the answer return exactly Unanswerable. '
          'Do not repeat the question, add an Answer: label, or continue with another question.')


def paper_text(p):
    return ('Title: '+p['title']+'\n\nAbstract: '+p['abstract']+'\n\n'+'\n\n'.join(
        (s['section_name'] or '')+'\n'+'\n\n'.join(s['paragraphs']) for s in p['full_text'])).rstrip()


def render(tok,paper,question=None):
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':'Paper:\n'+paper}]
    if question is not None: messages.append({'role':'user','content':'Question: '+question})
    text = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=question is not None,enable_thinking=False)
    encoded = tok(text,add_special_tokens=False,return_offsets_mapping=True)
    begin = text.index('Paper:\n')+len('Paper:\n'); end=begin+len(paper)
    assert text[begin:end]==paper
    body = [i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<b and a<end and b>begin]
    assert body and body==list(range(body[0],body[-1]+1)), 'Body contains unaligned special tokens'
    head,stop = body[0],body[-1]+1
    spans=[(max(0,a-begin),min(len(paper),b-begin)) for a,b in encoded['offset_mapping'][head:stop]]
    result={'prompt':text,'ids':encoded['input_ids'],'head':head,'stop':stop,'spans':spans}
    if question is not None:
        qb=text.rindex('Question: '+question)+len('Question: '); qe=qb+len(question)
        result['query_positions']=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<b and a<qe and b>qb]
        assert result['query_positions'] and min(result['query_positions'])>=stop
    return result


def paper_protocol(st,rt,paper,questions):
    source=render(st,paper)
    target=[render(rt,paper,q['question']) for q in questions]
    first=target[0]; prefix=first['ids'][:first['stop']]
    for t in target:
        assert t['head']==first['head'] and t['stop']==first['stop'] and t['ids'][:t['stop']]==prefix, 'Query changed evidence prefix'
    sb,tb=source['spans'],first['spans']
    bounds=sorted({0}|({b for a,b in sb if a<b}&{b for a,b in tb if a<b}))
    mapping={}; sp=tp=0
    for left,right in zip(bounds,bounds[1:]):
        si=[]; ti=[]
        while sp<len(sb) and sb[sp][1]<=right:
            a,b=sb[sp]
            if a<b and left<b: si.append(sp)
            sp+=1
        while tp<len(tb) and tb[tp][1]<=right:
            a,b=tb[tp]
            if a<b and left<b: ti.append(tp)
            tp+=1
        if not si or not ti: continue
        for rank,i in enumerate(ti):
            j=math.ceil((rank+1)*len(si)/len(ti))-1 if len(si)>len(ti) else rank*len(si)//len(ti)
            mapping[i]=source['head']+si[j]
    assert set(mapping)==set(range(len(tb))), 'No native evidence fallback permitted'
    return {'source_ids':source['ids'][:source['stop']], 'source_positions':[mapping[i] for i in range(len(tb))],
            'head_ids':prefix[:first['head']],'prefix_ids':prefix,'body_count':len(tb),
            'queries':[{'question_id':q['question_id'],'question':q['question'],
                'suffix_ids':t['ids'][t['stop']:], 'query_positions':t['query_positions'],
                'full_ids':t['ids']} for q,t in zip(questions,target)]}


def query_only(tok,question):
    text=tok.apply_chat_template([{'role':'system','content':SYSTEM},
        {'role':'user','content':'Question: '+question}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return list(tok.encode(text,add_special_tokens=False))
