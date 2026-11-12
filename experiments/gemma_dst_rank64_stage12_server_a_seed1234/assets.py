"""Download only public model artifacts; require exact server-B SHA256 identity."""
import concurrent.futures,hashlib,json,time,os
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text());DEST=Path(CFG['sender'])
EXPECTED=dict(line.split(None,1)[::-1] for line in '''eb5fd5e97ddd07b56778733e9653c07312529cb00980a318fc3e1c4e3b5a8f1f model-00001-of-00002.safetensors
fdde0e5aa5ced0fa203b3d50f4ab78168b7e3a3e08c6349f5cc9326666e1bb13 model-00002-of-00002.safetensors
50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946 added_tokens.json
fe16baf728db49457cde32802cd7efc0ac8a7a9877dbe22fe3322b2d9dc6ccd9 chat_template.json
9059f680f4dbd1957f35cb44b9fdd6948f4792db7a7a35ee353ea42e68adf7ff config.json
f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9 configuration.json
fd9324becc53c4be610db39e13a613006f09fd6ef71a95fb6320dc33157490a3 generation_config.json
77f4b67de084c31c7bcd373b039908108eee6c6181607e6d53da730e5f0bc659 model.safetensors.index.json
f688d6bb20c5017601c4011de7ca656da8485b540b05013efdaf986c0fcc918d preprocessor_config.json
3ffd5f11778dc73e2b69b3c00535e4121e1badf7018136263cd17b5b34fbaa53 processor_config.json
2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397 special_tokens_map.json
4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795 tokenizer.json
bfe25c2735e395407beb78456ea9a6984a1f00d8c16fa04a8b75f2a614cf53e1 tokenizer_config.json
1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c tokenizer.model'''.splitlines())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def fetch(name):
 path=DEST/name
 if path.exists() and sha(path)==EXPECTED[name]:return
 part=path.with_suffix(path.suffix+'.download')
 # An interrupted stream from B may already contain a correct weight prefix.
 if path.exists() and name.endswith('.safetensors') and not part.exists():path.rename(part)
 for attempt in range(5):
  offset=part.stat().st_size if part.exists() else 0
  try:
   url=f'https://modelscope.cn/models/LLM-Research/gemma-3-4b-it/resolve/master/{name}?download=true&offset={offset}'
   with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,90)) as response:
    response.raise_for_status()
    if offset and response.status_code==206:assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
    elif offset:offset=0
    expected=int(response.headers.get('Content-Length','0'));received=0;last=time.time()
    with part.open('ab' if offset else 'wb') as f:
     for chunk in response.iter_content(8*1024**2):
      f.write(chunk);received+=len(chunk)
      if time.time()-last>30:print(time.strftime('%F %T'),'DOWNLOAD',name,offset+received,flush=True);last=time.time()
    assert not expected or received==expected,(name,received,expected)
   digest=sha(part)
   if digest!=EXPECTED[name]:raise RuntimeError(f'Model checksum mismatch {name}: {digest}')
   part.replace(path);print('MODEL VERIFIED',name,flush=True);return
  except Exception as e:
   print('MODEL DOWNLOAD RETRY',name,attempt,str(e),flush=True)
   if attempt==4:raise
   time.sleep(5)
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(fetch,EXPECTED))
 v=dict(status='completed',source='Existing server-B Gemma exact SHA256; public original mirror',sha256=EXPECTED,time=time.strftime('%F %T'))
 (ROOT/'model_identity.json').write_text(json.dumps(v,indent=2)+'\n');print('ALL GEMMA ASSETS VERIFIED',flush=True)
if __name__=='__main__':main()
