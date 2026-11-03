"""Explicit final-answer scoring; legacy last-number extraction stays secondary."""
import re
from fractions import Fraction

NUMBER = r'[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:/[-+]?\d+)?'

def normalize(value):
    value=value.strip().replace(',', '').replace('$', '').replace('%', '')
    value=value.replace('\\,','').replace('\\!','').strip()
    value=re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}',r'\1/\2',value)
    if value.endswith('.') and not value.endswith('..'):value=value[:-1]
    if not re.fullmatch(NUMBER,value):return None
    try:return str(Fraction(value))
    except (ValueError,ZeroDivisionError):return None

def extract(text):
    boxes=[]
    for match in re.finditer(r'\\boxed\s*\{',text):
        start=match.end();depth=1
        for j in range(start,len(text)):
            depth+=(text[j]=='{')-(text[j]=='}')
            if depth==0:
                boxes.append(text[start:j]);break
    if boxes:
        value=normalize(boxes[-1]);return dict(prediction=value,answer_source='boxed',valid_final_answer=int(value is not None))
    marked=re.findall(r'(?:####|(?:final\s+answer|the\s+answer)\s*(?:is|:|=))\s*(.*)',text,re.I)
    if marked:
        value=normalize(marked[-1]);return dict(prediction=value,answer_source='explicit_marker',valid_final_answer=int(value is not None))
    value=normalize(text)
    return dict(prediction=value,answer_source='numeric_only' if value is not None else 'missing_or_malformed',valid_final_answer=int(value is not None))

def checks():
    examples=[(r'Compute 7+5. Final answer: 12.', '12'),(r'Work 1,234. \boxed{1,234}', '1234'),
        (r'\boxed{\frac{1}{2}}','1/2'),('The answer is -2.50.', '-5/2'),('#### 42','42'),
        ('42','42'),('First 10 then 20 and',None),(r'\boxed{42',None),(r'\boxed{wrong}',None),
        (r'\boxed{4} Actually \boxed{5}','5'),(r'\boxed{2/0}',None),(r'\boxed{42 dollars}',None)]
    for text,expected in examples:assert extract(text)['prediction']==expected,(text,extract(text))
    return dict(cases=len(examples),passed=True,arbitrary_last_number_not_primary=True)

if __name__=='__main__':print(checks())
