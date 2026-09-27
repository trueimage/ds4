"""Sequential fixed-prompt HTTP TTFT, with output and cache-image evidence.
Starts exactly one server; all sessions allocate at most 32768 tokens.
"""
import argparse, pathlib, subprocess, socket, urllib.request, urllib.error, time, json, signal, hashlib, statistics, os
p=argparse.ArgumentParser();p.add_argument('--checkout',type=pathlib.Path,required=True);p.add_argument('--model',required=True);p.add_argument('--prompts',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--staged',action='store_true');p.add_argument('--ssd',action='store_true');p.add_argument('--expect-continued-frontier',type=int);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
base=f'http://127.0.0.1:{port}';env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','MTL_'))}
if a.staged:env['DS4_KVSTORE_STAGE_PAYLOAD']='1'
cmd=['./ds4-server','-m',a.model,'--ctx','32768','--host','127.0.0.1','--port',str(port),'--kv-disk-dir',str(a.out/'kv'),'--kv-disk-space-mb','4096','--kv-cache-min-tokens','16','--kv-cache-cold-max-tokens','32768','--trace',str(a.out/'trace.log')]
if a.ssd:cmd+=['--ssd-streaming','--ssd-streaming-cache-experts','16GB']
(a.out/'command.json').write_text(json.dumps({'cwd':str(a.checkout),'argv':cmd,'staged':a.staged},indent=2))
def query(body):
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 start=time.perf_counter();first=None;last=None;itl=[];parts=[];usage=None;finish=None;done=False
 with urllib.request.urlopen(req,timeout=1200) as r:
  for line in r:
   if not line.startswith(b'data:'):continue
   data=line[5:].strip()
   if data==b'[DONE]':done=True;continue
   v=json.loads(data)
   if v.get('error'):raise RuntimeError(v)
   if v.get('usage'):usage=v['usage']
   choices=v.get('choices') or []
   if not choices:continue
   c=choices[0];d=c.get('delta') or {}
   if c.get('finish_reason'):finish=c['finish_reason']
   text=d.get('content') or d.get('reasoning_content')
   if text:
    now=time.perf_counter()
    if first is None:first=now
    else:itl.append((now-last)*1000)
    last=now;parts.append({k:d[k] for k in ['content','reasoning_content'] if d.get(k)})
 assert first and done and finish in ('stop','length') and usage, (first,done,finish,usage)
 assert usage['prompt_tokens']+usage['completion_tokens']<=32768,usage
 return {'ttft_ms':(first-start)*1000,'e2e_ms':(time.perf_counter()-start)*1000,'itl_ms':itl,'usage':usage,'parts':parts,'finish':finish}
with (a.out/'server.log').open('w') as log:
 proc=subprocess.Popen(cmd,cwd=a.checkout,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+300
  while True:
   assert proc.poll() is None,'server exited'
   try:
    with urllib.request.urlopen(base+'/v1/models',timeout=1) as r:model=json.load(r)['data'][0]['id'];break
   except (OSError,urllib.error.URLError):
    assert time.monotonic()<deadline,'startup timeout';time.sleep(.2)
  results=[]
  for path in sorted(a.prompts.glob('http-*.txt')):
   prompt=path.read_text();body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'think':False,'max_tokens':64,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
   result=query(body);result.update(name=path.stem+'-cold',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest());results.append(result)
   # Continue the exact sampled assistant text to exercise live reuse and continued stores.
   answer=''.join(x.get('content','') for x in result['parts'])
   body['messages'] += [{'role':'assistant','content':answer},{'role':'user','content':'Continue with one additional sentence.'}]
   continued=query(body);continued.update(name=path.stem+'-continued');results.append(continued)
   if a.expect_continued_frontier and path.stem=='http-10240':assert continued['usage']['prompt_tokens']==a.expect_continued_frontier,continued['usage']
   (a.out/'results.json').write_text(json.dumps(results,indent=2))
   print(path.name,round(result['ttft_ms'],3),round(continued['ttft_ms'],3),flush=True)
 finally:
  if proc.poll() is None:
   proc.send_signal(signal.SIGINT)
   try:proc.wait(timeout=60)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
 assert proc.returncode==0,proc.returncode
if a.expect_continued_frontier:
 assert any(f'kv cache stored tokens={a.expect_continued_frontier} ' in line and 'reason=continued' in line for line in (a.out/'server.log').read_text().splitlines()), 'continued frontier was not stored'
assert not list((a.out/'kv').glob('*.tmp.*')), 'unfinished checkpoint temporary file'
images=[]
for f in sorted((a.out/'kv').glob('*.kv')):
 with f.open('rb') as fp:
  header=bytearray(fp.read(48));header[24:40]=b'\0'*16;h=hashlib.sha256(header)
  while block:=fp.read(8<<20):h.update(block)
 images.append({'name':f.name,'bytes':f.stat().st_size,'masked_sha256':h.hexdigest()})
(a.out/'images.json').write_text(json.dumps(images,indent=2))
assert images,'no checkpoints were stored'
